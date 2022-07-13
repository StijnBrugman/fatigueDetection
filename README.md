# Fatigue Recognition using Compter Vision Technology
This repository will contain all software and instructions for the development of a fatigue recognition system. This project was made for the BSc thesis of Creative Technology at the University of Twente.

## The thesis abstract
Fatigue is a major determinant in traffic accidents. Normalization and automation of modern vehicles only increase the urgency for the development of an advanced driver assistant system (ADAS) that reliably can detect the driver’s fatigue state. In this thesis, an ADAS is proposed based on the Viola-Jones Algorithm, and drowsiness metric. Haar-like feature-based cascade classifiers are combined with AdaBoost to locate the face and then extract relevant landmarks. The eye ratio aspect (EAR) is determined from the obtained landmarks. Contrary to conventional methods, a peak detection function is used on the EAR sequence for the identification of blinking patterns. Subsequently, the classification of the drivers uses the percentage of eyelid closure (PERCLOS), blink frequency, and entropy to classify the fatigue state. The proposed system is evaluated using a metric evaluation on the Eyeblink8 dataset, achieving a satisfactory level of precision (0.839%) and recall (0.893%). Additionally, the user evaluation demonstrated the state-of-the-art and real-time performance accomplished by the proposed system.

### Software & Requirements
The software is fully written in python 3.10.0.
A little intro about running the program. First install the requirements by runnning this command.
```
$ pip install -r requirements.txt
```
or this.
```
$ python -m pip install -r requirements.txt
```
Now you can run the program.
```
$ python ./main.py
```

## Flags
Several flags can be added to the command to provide additional functionality.
'''
$ python ./main.py -flags
'''
flags:
* -h : -help | Provides an overview of all the parsing options
* -c : -cam | Views the camera capture frame
* -s : -save | Safe the obtained data to external file
* -v : -vis | Visualizes the data being analyzed

By default the compiler searches files to include in ./ and ./std/. You can add more search paths via the '-I' flag before the subcommand: ./porth -I <custom-path> com .... See ./porth help for more info.

## Contact
For more information about the project, you can contact Stijn Brugman ([s.r.d.brugman@student.utwente.nl](mailto:s.r.d.brugman@student.utwente.nl)).

## Special thanks
The development of my thesis would not have been possible if it had not been for the support of my supervisor. I am very grateful for the support, knowledge, and expertise provided by Job Zwiers. J. Zwiers guided me through the sometimes rough process of this research and taught me how to construct a thesis in a proper manner. Additionally, this endeavor would not have been possible without the assistance of Fleur Bake, who provided this challenging thesis and provided the required resources for me to develop the thesis. Furthermore, I want to thank Daniel Davison for the constructive feedback provided during the development of my thesis. In addition, a special thanks should also go to all the participants that helped with the evaluation of my thesis. Lastly, I would be remiss in not mentioning the Line-Up. Their belief and supportiveness during this process kept my motivation very high through this process.