# ICOW Service Install

For a quick install, use the [sandbox](/sandbox)

## Helm Chart
See [Installing Helm](https://helm.sh/docs/intro/install/) for instructions on installing Helm

```
git clone git@github.com:Striveworks/LIteCOW.git
cd LIteCOW
helm install -n icow --create-namespace icow .
```

### Enable GPU support

The helm chart can be used to install LIteCOW with gpu support enable as well. To enable this feature make the changes suggested in the comments of the values.yaml file.
