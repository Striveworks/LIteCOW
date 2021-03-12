# ICOW Service Install

For a quick install, use the [sandbox](/sandbox)

## Helm Chart
See [Installing Helm](https://helm.sh/docs/intro/install/) for instructions on installing Helm

```
git clone git@nacho.striveworks.us:chariot/icow-light.git
cd icow-light
helm install -n icow --create-namespace icow .
```
