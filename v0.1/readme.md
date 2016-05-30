## Tutorial

Here is a small tutorial on how to run the example provided with the code.

First a warning: running this code will not be straight forward. If you have questions, write to me: jahuth(ät)uos.de

```
cd v0.1
python gui.py
```
The gui might only work for certain Qt and IPython installations.
You might have to dive into the first few lines of gui.py and misc_guy.py to remedy differences on your system.
Eg. you can use the `use_qt4` variable to switch between PyQt4 and PySide.
Also you might get warnings if you installed jupyter rather than the old IPython. In that case, substitute

```
from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
```
with
```
from qtconsole.rich_jupyter_widget import RichJupyterWidget as RichIPythonWidget
```

*If* you get it to run (if not, do not hesitate to send me a mail), you will be greeted with a window separated into three parts.

![The main interface](https://raw.githubusercontent.com/jahuth/scheduling_and_aggregating_numerical_simulations/master/v0.1/demo_images/gui000.png "The main interface")

The middle section is a file browser. Select the Example_Experiment.py file by clicking on it.
If the file is selected, the right side should now contain two Buttons. Each of them corresponds to a function exposed by the Experiment to create Sessions.
When one of the buttons is clicked, a Dialog opens to let you change the default parameters. On closing this window, a new Session will be created and opened.

The other two sections contain the recently used Sessions: on the left globally, on the right for the selected experiment.

![The main interface](https://raw.githubusercontent.com/jahuth/scheduling_and_aggregating_numerical_simulations/master/v0.1/demo_images/gui001.png "The main interface")
![The main interface](https://raw.githubusercontent.com/jahuth/scheduling_and_aggregating_numerical_simulations/master/v0.1/demo_images/gui002.png "The main interface")
![The main interface](https://raw.githubusercontent.com/jahuth/scheduling_and_aggregating_numerical_simulations/master/v0.1/demo_images/gui003.png "The main interface")

### The Session Panel

Each session is opened in its own tab. In the session tab, there are three more tabs, one for the Task Overview and two ipython consoles (one for results, the other one for debugging).

![The Session interface](https://raw.githubusercontent.com/jahuth/scheduling_and_aggregating_numerical_simulations/master/v0.1/demo_images/gui004.png "The Session interface")

To run the experiment, either click on the large "Run" button to execute one task, or click on "Autorun OFF" button to change it to "Autorun ON". This will start the next task once one is finished.

![The Session interface](https://raw.githubusercontent.com/jahuth/scheduling_and_aggregating_numerical_simulations/master/v0.1/demo_images/gui005.png "The Session interface")

![The Session interface](https://raw.githubusercontent.com/jahuth/scheduling_and_aggregating_numerical_simulations/master/v0.1/demo_images/gui006.png "The Session interface")

Each task will have a random sleep call, resulting in an jumpy progress.

### The Task Panel

When clicking on a specific task, the description, log and data output of the task can be inspected.
The Data tab of each task contains an HTML view of the data container.

![The Task interface](https://raw.githubusercontent.com/jahuth/scheduling_and_aggregating_numerical_simulations/master/v0.1/demo_images/gui007.png "The Task interface")

![The Task interface in the data tab](https://raw.githubusercontent.com/jahuth/scheduling_and_aggregating_numerical_simulations/master/v0.1/demo_images/gui008.png "The Task interface in the data tab")


### The Results Panel

The results tab consists of an IPython console (that already contains a reference to the sessions as `silver_session`) and on the right side buttons that provide code from the experiment file itself.
For this example there are two pieces of code that both load all the generated random values and then produce a plot.

![The Results interface](https://raw.githubusercontent.com/jahuth/scheduling_and_aggregating_numerical_simulations/master/v0.1/demo_images/gui009.png "The Results interface")

The code for each of the two actions is:

```
v = silver_session.root_task.get_data(['values_1','values_2'])
for vv in v:
    plot(vv[0],vv[1],'.')
```

and 

```
v = silver_session.root_task.get_data(['values_1','values_2'])
plot([np.mean(vv[0]) for vv in v])
plot([np.mean(vv[1]) for vv in v])
```

`get_data` fetches all the results from the respective Task Data Containers.
