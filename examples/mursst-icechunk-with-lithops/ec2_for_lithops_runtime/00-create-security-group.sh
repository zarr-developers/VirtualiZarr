export SG_GROUP_NAME=XXX
export VPC_ID=XXX
aws ec2 create-security-group --group-name $SG_GROUP_NAME --description "security group for lithops runtime builder ec2" --vpc-id $VPC_ID
