# SSCP19 Project 7 - LV Mechanics

## Docker
We have made a prebult image with all the necessary requirements
installed. You can get it by typing


Build docker image
```
docker pull finsberg/sscp19_project7
```
Run container (one level up `cd ..`)
```
docker run -ti --name summer-school-container -e "TERM=xterm-256color" -w /home/fenics/shared -v $(pwd):/home/fenics/shared finsberg/sscp19_project7
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
