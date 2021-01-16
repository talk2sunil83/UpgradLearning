
# coding: utf-8

# # <center>Working With Plotly</center>

# <img src="plotly.png">

# In[1]:


#import
import plotly
from plotly import __version__
print (__version__)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# In[2]:


#Jupyter setup
init_notebook_mode(connected=True)


# <br>
# 
# To save the chart to Plotly Cloud or Plotly Enterprise, use `plotly.plotly.iplot`.
# 
# Use py.plot() to return the unique url and optionally open the url.
# 
# Use py.iplot() when working in a Jupyter Notebook to display the plot in the notebook.
# 
# <br>

# In[3]:


iplot([{"x": [1, 2, 3, 4, 5], "y": [1, 2, 6, 4, 1]}])


# In[4]:


import plotly.graph_objs as go
import numpy as np

x = np.random.randn(2000)
y = np.random.randn(2000)
iplot([go.Histogram2dContour(x=x, y=y, contours=dict(coloring='heatmap')),
       go.Scatter(x=x, y=y, mode='markers', marker=dict(color='white', size=3, opacity=0.3))], show_link=False)


# # <center>Episode 2</center>

# In[40]:


import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np


# In[45]:


N = 100
example_x = np.random.rand(N)
example_y = np.random.rand(N)



# In[46]:


trace = go.Scatter(
    x = example_x,
    y = example_y,
    mode = 'markers')


# In[47]:


data = [trace]
iplot(data)


# In[ ]:


#PIE CHART CREATION


# In[51]:


#Expenses

#breakdown each category 
groups = ['Rent','Food','Bills','Miscellaneous']
#create amount
amount = [1000,518,331,277]
#style
colors = ['#d32c58', '#f9b1ee', '#b7f9b1', '#b1f5f9']

trace = go.Pie(labels=groups, values=amount,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=25),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=3)))

#plot - 
iplot([trace])

