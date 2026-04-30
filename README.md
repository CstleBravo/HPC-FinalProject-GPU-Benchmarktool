# HPC-FinalProject-GPU-Benchmarktool
Simple benchmark tool that uses torch and runs a vector, matrix multiplication, and a few other tests.
At this current point in time there are two versions.

The first version is a basic cli program that needs to be run with python installed on the system: with the files of gpubenchmark.py, gui.py, requirements.txt. With python installed in the environment you can run "pip install -r requirements.txt" which should install all libraries that are needed. Then afterwards you can run either "python gpubenchmark.py" or "python gui.py" to either have a basic CLI version or a general gui version.

The other version is a portable build that has all the instructions under the .md file and the batch file. This will create a .exe file that can be put on any windows machine and run just fine. WARNING: this will install the entire torch library making the current program despite it being fairly light and easy to run a whopping 10 GBs, I am looking into compacting this but at the current point and time I have no idea how to do that correctly. 
