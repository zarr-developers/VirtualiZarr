# look up the group id created
export SECURITY_GROUP_ID=XXX
export YOUR_IP=$(curl -s https://checkip.amazonaws.com)
export AMI_ID=ami-027951e78de46a00e
export SSH_KEY_NAME=XXX
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --ip-permissions '{"IpProtocol":"tcp","FromPort":22,"ToPort":22,"IpRanges":[{"CidrIp":"'$YOUR_IP'/32"}]}'
aws ec2 run-instances --image-id $AMI_ID \
  --instance-type "t3.medium" --key-name $SSH_KEY_NAME \
    --block-device-mappings '{"DeviceName":"/dev/xvda","Ebs":{"Encrypted":false,"DeleteOnTermination":true,"Iops":3000,"SnapshotId":"snap-01783d80c688baa0f","VolumeSize":30,"VolumeType":"gp3","Throughput":125}}' \
    --network-interfaces '{"AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["'$SECURITY_GROUP_ID'"]}' \
    --credit-specification '{"CpuCredits":"unlimited"}' \
    --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required"}' \
    --private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":true,"EnableResourceNameDnsAAAARecord":false}' \
    --count "1"
