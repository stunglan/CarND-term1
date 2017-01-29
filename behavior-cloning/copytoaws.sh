#!/bin/bash

scp -i "~/Google Drive/certificates/deeplearningcertificate.pem"  $1 ubuntu@ec2-54-194-185-145.eu-west-1.compute.amazonaws.com:~/src/github/CarND-term1/behavior-cloning
