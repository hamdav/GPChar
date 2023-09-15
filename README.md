# User guide

## Purpose

### The problem
The problem this program is trying to solve is to investigate how some quantity depends on a number of other quantities, where the number of evaluations is limited. It might be limited because one evaluation takes a long time to do, either because it's a heavy calculation, or because it involves a human preforming some measurement in the real world; or it might be limited simply because you have gotten some data already and cannot get more. In both of these cases you would like to get the best possible approximation of the true relationship between the quantities, and also an estimate of the uncertainty in your approximation. In the case where you can acquire more data, you would also like to do this at the points that will get you the most information, so that you don't waste evaluations at points where you already know the outcome. Additionally, once a good approximation of the relationship is obtained, you would want some way of visualizing this which isn't trivial when you have more than 2 input quantities. Summarizing, we want something that can:
- approximate the relationship between n inputs and m outputs from a limited number of datapoints, and give the uncertainty of it's estimation,
- visualize this relationship in a useful manner, and
- suggest what combination of input values that should be evaulated next to maximize gained information.

#### Example

One example could be to investigate how the eigenfrequency of a photonic crystal unit cell depends on the geometric parameters like width, length and hole radii, as well as wavevector. Since each evaluation takes around a minute, and the parameter space is 5 dimensional, it is important to choose the sampling points carefully and extract as much information as possible from each point.

### The solution

Just like the problem, the solution comes in three parts.
- To approximate the relationship, a Gaussian Process Regressor is used.
- To visualize the relationship, a web app with sliders for each dimension is used.
- To obtain the point where new data should be taken, check where the uncertainty of the Gaussian Process Regressor is.

TODO: Add screenshots.


## Installation

- Clone the repo
- Go to the clone, make sure you are in a conda environment you want to be, and run `conda install numpy scipy pandas scikit-learn plotly dash`.*
- run `pip install --user -e .`  (mind the dot).

*You could just run the `pip install --user -e .` directly and it will install the dependencies, but it will do so through pip, which is bad practice… one should install as much as possible with conda, and only that which you can’t with pip because conda is better at managing versions and dependencies.

## Usage

There are two possible ways of using it. Either you have a function that you can evaluate at points, and you want to know how that function behaves. Or you had a function and you just want to fit a gaussian process to your data. The process is very similar either way, though if you just want to fit to data, you may skip the function-defining step.

### Creating the GPChar

- import the necessary things:

```python
from gpchar.gpchar import GPChar
from gpchar.graphics import launch_dash_app
```

- Define the function. It needs to take a numpy array as inputs and spit out a numpy array of the outputs. If it doesn’t, just write a small wrapper function that does. Below I have an example

```python
def simulate_unitcell(a, w, r1, r2, k):
	  model = client.load('LN_unitcell.mph')
    model.parameter('GP_a', f'{a}[nm]')
    model.parameter('GP_w', f'{w}[nm]')
    model.parameter('GP_r1', f'{r1}')
    model.parameter('GP_r2', f'{r2}')
    model.parameter('GP_k', f'{k} * pi / GP_a')
    
    model.mesh()
    model.solve()
    
    result = model.evaluate(['ewfd.freq', 'ewfd.Qfactor'])
    
    client.remove(model)

    # The mode we're interested in is the lowest one (which should be the one with the highest Q)
    return np.array([result[0][0], np.log10(result[0][1]),
	                   result[1][0], np.log10(result[1][1])])

def f(x):
    return simulate_unitcell(*x)
```

- Define the bounds for the arguments. Where do you want to investigate the function? Put these into a list of tuples

```python
bounds = [(300,700), (1000,2000), (0.1, 0.4), (0.1, 0.4), (0.5, 1.0)]
```

- That’s it! Now you can create an instance of the GP characterizer class like below. The 5 is for the input dimension and 4 for the output. `datafile.csv` is the file that function evaluations are saved to, and which may contain already evaluated points to start from. When your function takes a long time to evaluate, it’s always good to keep a persistent record.

```python
gpc = GPChar(f, bounds, 5, 4, "datafile.csv")
```

### Visualization

Running the command below will launch an app, with a link that you can press opening the app in your web browser. This call will be blocking though (meaning it will not return) so if you want to evaluate functions in parallel, launch it in it’s own thread

```python
# Strings that will appear in the menus
input_names = ["a", "w", "r1", "r2", "k"]
output_names = ["f1", "Q1", "f2", "Q2"]

# Normal launch (blocking)
launch_dash_app(gpc, bounds, input_names, output_names)
print("This would not print")

# Launch in separate thread
app_thread = launch_dash_app_in_thread(gpc, bounds, input_names, output_names)
print("This would print")

```

### Acquiring function evaluations

Without data, the visualization won’t do much good. There are two ways of gathering data, and each can be done in a thread or not. The first is just acquiring evaluations of random points inside the bounds.

The second finds the point where the standard deviation of the GP is the largest and acquires a function evaluation at this point. Both can be repeated as necessary.

These are thread safe, meaning that they can be run at the same time as the dash app and each other, however running them at the same time as each other is probably not a good idea since the two comsol clients will be fighting each other for resources.

Before there are any datapoints, uncertainty is maximal everywhere, so it’s probably a good idea to do ~20 or so random evaluations. How many you want depends on the dimensionality of the space, just pulling a number out of my ass I’d say like 4 times the input dimension.

I do something like this, killing the program with `ctrl+c` when I feel like I’m done.

```python
print("Performing random evaluations")
thread = gpc.acquire_random_evaluations_in_thread(20)
thread.join() # <- waits for thread to exit

print("Performing smart evaluations")
thread = gpc.acquire_evaluations_in_thread(1000000)
thread.join()
```

### Advanced visualization

Relegated to the bottom, but if you want to write your own visualizations, you can get GP predictions with the functions `get_1d_prediction` and `get_2d_prediction` to get raw prediction + uncertainties. The plot it however you like. You can of course also access the GaussianProcessRegressor objects and use those directly if you like. Look at the source!

# Tips & Tricks

## Reuse data!

While using this script, I found that I first guessed that I wanted to investigate data in the range of
- a: 300nm to 700nm
- w: 1000nm to 1500nm
- r1 and r2: 0.1 to 0.4
- k: 0.5 to 1.0

But then I realized that I probably wanted to check what was happening at larger w, up to 2000nm, aswell. Since the data that I had already taken was not invalidated by this, any datapoints there are still useful, I just cancelled the computation (remember that all of the datapoints are stored in a file) and restarted with new bounds. 

Moreover, I later realized that I wanted to see how an additional parameter h (the thickness of the cell, so far fixed to 220 nm) influenced the frequency, and also that unit cells with a > 500 were bad so I didn't want to investigate them further. To do this, I went into the file with the datapoints and added a column between r2 and k and set the value to 220 nm for all of the datapoints and voila, when I rerun the GPChar it will start sampeling points of different h, but it will still have learnt all of the things about the other dimensions at h=220nm, so it isn't starting from 0. In fact, with only very few additional datapoints, it got a very good approximation for how h affects the frequency for *all* of the different combinations of values for the other dimensions. Note also that there was no need to remove the points where a > 500, even though I set the bounds to be 300nm to 500nm now, the GP might as well learn from those datapoints as well, I just didn't want it to waste time gathering more evaluations there.

