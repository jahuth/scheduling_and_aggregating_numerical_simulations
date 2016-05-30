#!silver_gui
import numpy
import pandas
from matplotlib.pylab import *
import scipy
import numpy as np
import glob

import silver.schedule as schedule

description = """
This Experiment creates some random numbers
"""

def createTasks(name="unnamed", 
                mean1_a =  [0,10,20,30], 
                sigma1_a = [0,0.1,0.5,1.0],  
                mean2_a =  [0,10,20,30],  
                sigma2_a=  [0,0.1,0.5,1.0], 
                trials=1, experiment = None,**kwargs):
    task_directory = 'Tasks/'+name+''
    import os
    if not os.path.exists(task_directory):
        os.makedirs(task_directory)
    session = schedule.Session(experiment, task_directory+'/Session')
    main_task = schedule.Task("Main Task", parallel=True, path=task_directory+"/.",tags=['main_task'])
    main_task.variables['task_directory'] = task_directory
    main_task.variables['mean1_a'] = mean1_a
    main_task.variables['sigma1_a'] = sigma1_a
    main_task.variables['mean2_a'] = mean2_a
    main_task.variables['sigma2_a'] = sigma2_a
    session.root_task = main_task
    for mean1 in mean1_a:
        for mean2 in mean2_a:
            mean_task = schedule.Task("Mean values: "+str(mean1)+" and "+str(mean2),main_task,variables = {'mean1':mean1,'mean2':mean2})
            main_task.subtasks.append(mean_task)
            for sigma1 in sigma1_a:
                for sigma2 in sigma2_a:
                    t = schedule.Task("Standard Deviations: "+str(sigma1)+" and "+str(sigma2),mean_task,
                        cmd = schedule.Command("""
import numpy as np
from numpy import deg2rad

with task.data as data:
    from matplotlib.pylab import *
    values_1 = mean1 + sigma1 * np.random.randn(100)
    values_2 = mean2 + sigma2 * np.random.randn(100)
    plot(values_1,values_2,'k.')
    data.savefig('Raster',gcf())
    data.set('values_1',values_1)
    data.set('values_2',values_2)
    time.sleep(randint(0,2)) # sleep between 0 and 2 seconds
"""),
                    variables = {'sigma1':sigma1, 'sigma2':sigma2 },
                    parallel=True,tags=['stimulus'])
                    mean_task.subtasks.append(t)
    return session

actions = {
        'Create default means': {'function':createTasks,'kwargs':{ 'name':"random_numbers_default", 
                'mean1_a'  : [0,10,20,30], 
                'sigma1_a' : [0,0.1,0.5,1.0],  
                'mean2_a'  : [0,10,20,30],  
                'sigma2_a' : [0,0.1,0.5,1.0]
                }
            },
        'Create small means': {'function':createTasks,'kwargs':{ 'name':"random_numbers_small", 
                'mean1_a'  : [0,1,2,3], 
                'sigma1_a' : [0,0.1,0.5,1.0],  
                'mean2_a'  : [0,1,2,3],  
                'sigma2_a' : [0,0.1,0.5,1.0]
                }
            }
    }

task_context_actions = {
    }

result_actions = {
    'Plot means' : {'function':None, 'description':'', 'scipy':
"""
v = silver_session.root_task.get_data(['values_1','values_2'])

plot([np.mean(vv[0]) for vv in v])
plot([np.mean(vv[1]) for vv in v])

"""}, 'Plot all raster plots' : {'function':None, 'description':'', 'scipy':
"""
v = silver_session.root_task.get_data(['values_1','values_2'])
for vv in v:
    plot(vv[0],vv[1],'.')
"""}

}

