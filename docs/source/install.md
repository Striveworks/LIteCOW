# ICOW Service Install

For a quick install, use the [sandbox](/sandbox)

## Helm Chart
Before installing with helm you'll need to have a working knative-serving installation. We use kourier as our networking layer.
[Instructions](https://knative.dev/docs/install/any-kubernetes-cluster/) to set this up can be found on knative's website. 

See [Installing Helm](https://helm.sh/docs/intro/install/) for instructions on installing Helm

Once you have a working knative serving installation and helm installation the following will install litecow:
```
git clone git@github.com:Striveworks/LIteCOW.git
cd LIteCOW
helm install -n icow --create-namespace icow deployment/icow
```

### Enable GPU support

The helm chart can be used to install LIteCOW with gpu support enable as well. To enable this feature make the changes suggested in the comments of the values.yaml file.
