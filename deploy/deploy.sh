#!/bin/bash
# Deploy Worldline to AWS: bundle the app and processed data, upload to a private S3 bucket,
# then create or update the CloudFormation stack. Re-running ships a new bundle to the
# running instance through SSM (no SSH).
#
# Usage: deploy/deploy.sh            (needs the AWS CLI signed in; region from AWS_REGION or config)
set -euo pipefail
cd "$(dirname "$0")/.."

STACK=worldline
REGION=${AWS_REGION:-$(aws configure get region)}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=worldline-$ACCOUNT-$REGION
VERSION=$(git rev-parse --short HEAD)$([ -z "$(git status --porcelain)" ] || echo -dirty)
KEY=bundles/worldline-$VERSION-$(date +%Y%m%d%H%M%S).tar.gz

if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "creating bucket $BUCKET"
  if [ "$REGION" = us-east-1 ]; then aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  else aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null; fi
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
fi

# The bundle is every file git tracks or would track (so uncommitted edits ship too),
# plus the processed clip data, source clips and detector model, which git ignores.
TMP=$(mktemp -d)
{ git ls-files -co --exclude-standard; find web/data clips models -type f -not -path 'web/data/_agent/*' -not -path 'web/data/_uploads/*'; } | sort -u > "$TMP/files"
tar -czf "$TMP/code.tar.gz" -T "$TMP/files"
aws s3 cp "$TMP/code.tar.gz" "s3://$BUCKET/$KEY" --only-show-errors
rm -rf "$TMP"
echo "uploaded s3://$BUCKET/$KEY"

PREFIX_LIST=$(aws ec2 describe-managed-prefix-lists --filters Name=prefix-list-name,Values=com.amazonaws.global.cloudfront.origin-facing \
  --query 'PrefixLists[0].PrefixListId' --output text)

if aws cloudformation describe-stacks --stack-name "$STACK" >/dev/null 2>&1; then
  # Apply template changes (IAM, CloudFront) but keep the original BundleKey: changing it would
  # change the instance's user data and restart it with a new address, breaking the CDN origin.
  CURRENT_KEY=$(aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].Parameters[?ParameterKey=='BundleKey'].ParameterValue" --output text)
  aws cloudformation deploy --stack-name "$STACK" --template-file deploy/worldline.yaml \
    --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
    --parameter-overrides BundleBucket="$BUCKET" BundleKey="$CURRENT_KEY" CloudFrontPrefixList="$PREFIX_LIST"
  INSTANCE=$(aws cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text)
  echo "updating running instance $INSTANCE"
  CMD=$(aws ssm send-command --instance-ids "$INSTANCE" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sed -i 's#^BUNDLE_KEY=.*#BUNDLE_KEY=$KEY#' /etc/worldline.env\",\"/usr/local/bin/worldline-update\"]" \
    --timeout-seconds 1800 --query Command.CommandId --output text)
  # rebuilding the image takes several minutes; poll until the command finishes
  while STATUS=$(aws ssm get-command-invocation --command-id "$CMD" --instance-id "$INSTANCE" --query Status --output text 2>/dev/null); \
        [ "$STATUS" = Pending ] || [ "$STATUS" = InProgress ] || [ "$STATUS" = Delayed ] || [ -z "$STATUS" ]; do sleep 15; done
  echo "update: $STATUS"
  [ "$STATUS" = Success ]
else
  aws cloudformation deploy --stack-name "$STACK" --template-file deploy/worldline.yaml \
    --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
    --parameter-overrides BundleBucket="$BUCKET" BundleKey="$KEY" CloudFrontPrefixList="$PREFIX_LIST"
fi
aws cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs" --output table
