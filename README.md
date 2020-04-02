# Particle-Swarm-Optimization-PSO-
Particle Swarm Optimization (PSO)
Particle Swarm Optimization (PSO) is one of the heuristic optimization methods that use swarming rules of
the birds/insects that we see in nature. The main idea is to follow the leading member/leading group of
particles of the swarm who are close to the goal (most probably the food) of the team.
The very first thing to do is to import neccesarry modules to our python environment. The pygame moodule is
used for visualization of what is happening through the iterations. It can be taken to "off" if not necessary.
# Swarm Class
The swarm class given below consists of sub-routines what is needed for PSO.
The init function is the main body of our class where we define the basic features of the swarming particle
objects. We input the function as well as the lower and upper bounds that are obligatory. Some ("egg" and
"griewank") of the universal test functions defined in the "Universal Test Functions" section given after, are
predefined inside the init function with their lower and upper boundaries, however, in real cases, these
arguments should be input by the user. The pygame screen where the user can visualize what is happening
inside the swarm is in the "off" mode by default but can be made "on" by making the "display" argument
"True". The migration is inherently on if nothing is done but can be closed by changing the "migrationexists"
argument to "False". The migration probability is set to 0.15 by default and can be changed if necessary. The
default number of particles inside the swarm is kept at 50 but can be changed, either.
The functions "coefficients", "evaluate_fitness", "distance", "migration" and "optimize" are the functions where
the main PSO algorithm is run that the user does not need to consider about. The values given in the
coefficients were obtained from MoGenA (Multi Objective Genetic Algorithm) and should be kept unchanged
for the best performance, that is why any user calling swarm class will not see the values in normal usage.
There will be migration with a probability of 15%, where the worst 20% of the swarm is going to be changed
with new particles. Some of these new particles will be placed to the domain randomly while the rest will be
placed according to some predefined algorithm that is promoting the best particles. This migration algorithm
deveoped here increases the rate of finding the global optimum value by decreasing the chance of being
stuck at a alocal minima.
The itarations are done by calling "update" function where the default iteration number is given as 50.
# Universal Test Functions
Universal test functions are used to evaluate/compare the performance of the optimization methods. They
are generally very hard functions to optimize with lots of local minima that are very close to global minima.
The mostly used ones are given in the address https://www.sfu.ca/~ssurjano/optimization.html
(https://www.sfu.ca/~ssurjano/optimization.html) and some of them given in the next section, below are
deployed inside pso module.
# Generate Particles
The first thing to do is to generate the swarming particles randomly inside the boundaries of the function
domain. The example below is done with "egg" function predefined inside the swarm class with its lower
[-512,-515] and upper [512, 512] boundaries. The number of particles is given as 100, which means we will
have a swarm that has 100 particles inside. The default display mode is off but since we want to visualize
what is happening, the display mode changed to "True". There will be migration in the swarm and its
probability parameter is kept as default.
The "egg" function is given as follows.
The global optimum point is at (512,404.2319) with a value of -959.6407.
