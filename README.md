# SSCP19 Project 7 - LV Mechanics

## Running docker container


Build docker image
```
docker build -t summer-school-image .
```
Run container (one level up `cd ..`)
```
docker run -ti --name summer-school-container -e "TERM=xterm-256color" -w /home/fenics/shared -v $(pwd):/home/fenics/shared summer-school-image
```

Check out [docker_workflows](https://github.com/ComputationalPhysiology/docker_workflows/) for more example of how to use docker in
your workflow.

### Requirements for LHS
pyDOE
scipy

### Requirements for PCA
matplotlib
sklearn
numpy
seaborn