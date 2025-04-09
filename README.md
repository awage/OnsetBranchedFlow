# OnsetBranchedFlow


The code is under MIT License, please cite the this repository or the 
companion article to this repo. 

Alexandre Wagemakers

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add(DrWatson) # install globally, for using `quickactivate`
   julia> Pkg.activate(path/to/this/project)
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.


