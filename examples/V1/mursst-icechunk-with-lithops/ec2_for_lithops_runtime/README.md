# Launch and use an EC2 for building the Lithops lambda runtime
## Note: This example uses a pre-2.0 release of VirtualiZarr

The scripts in this directly will help to launch and set up an ec2 so that you can build and push a lithops lambda runtime.

You will need AWS console and CLI access.

Steps:

1. Access the AWS console to create an SSH key in AWS that you can associate the EC2 when launching.
2. Add a `SECURITY_GROUP_NAME` of your choosing and appropriate `VPC_ID` to `00-create-security-group.sh` and execute that script.
3. Add the `SECURITY_GROUP_ID` and other required variables to `01-launch-ec2.sh` and execute that script.
4. Add the `INSTANCE_ID` to `02-setup-ec2-role.sh` and execute that script.
5. SSH into the instance and execute the scripts in `03-setup-ec2.sh`.
