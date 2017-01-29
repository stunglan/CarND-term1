≈#!/bin/bash

mv model.json model.json.old
scp -i "~/Google Drive/certificates/deeplearningcertificate.pem"  ubuntu@ec2-54-194-185-145.eu-west-1.compute.amazonaws.com:~/src/github/CarND-term1/behavior-cloning/model.json .
