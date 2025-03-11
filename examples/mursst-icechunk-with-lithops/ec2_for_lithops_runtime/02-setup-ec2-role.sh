#!/bin/bash

# Set variables
ROLE_NAME="EC2_Lithops_Lambda_Builder"
INSTANCE_ID=XXX  # Replace with your EC2 instance ID
POLICY_NAME="EC2LithopsLambdaPolicy"
REGION=XXX

# Step 1: Create the IAM role
aws iam create-role --role-name $ROLE_NAME \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": { "Service": "ec2.amazonaws.com" },
                "Action": "sts:AssumeRole"
            }
        ]
    }' > /dev/null

echo "✅ IAM Role '$ROLE_NAME' created."

# Step 2: Attach necessary policies
aws iam put-role-policy --role-name $ROLE_NAME --policy-name $POLICY_NAME \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:CompleteLayerUpload",
                    "ecr:UploadLayerPart",
                    "ecr:InitiateLayerUpload",
                    "ecr:PutImage",
                    "ecr:BatchGetImage",
                    "lambda:CreateFunction",
                    "lambda:UpdateFunctionCode",
                    "s3:GetObject",
                    "s3:ListBucket",
                    "ecr:CreateRepository"
                ],
                "Resource": "*"
            }
        ]
    }' > /dev/null

echo "✅ IAM policy attached to role '$ROLE_NAME'."

# Step 3: Create an Instance Profile and associate with the role
aws iam create-instance-profile --instance-profile-name $ROLE_NAME > /dev/null
aws iam add-role-to-instance-profile --instance-profile-name $ROLE_NAME --role-name $ROLE_NAME

echo "✅ Instance profile '$ROLE_NAME' created and role attached."

# Step 4: Attach the IAM role to the running EC2 instance
aws ec2 associate-iam-instance-profile --instance-id $INSTANCE_ID \
    --iam-instance-profile Name=$ROLE_NAME > /dev/null

echo "✅ IAM role '$ROLE_NAME' attached to instance '$INSTANCE_ID'."

# Step 5: Confirm the role is attached
echo "🔄 Waiting for role to be active..."
sleep 10
aws ec2 describe-instances --instance-ids $INSTANCE_ID --query "Reservations[*].Instances[*].IamInstanceProfile" --output json

echo "✅ Done! The EC2 instance now has the necessary permissions."
