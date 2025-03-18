from control.GymEnv import GymEnv
import matplotlib.pyplot as plt

def d(g):
    x,y = g.cutter._compute_relative_position()
    d = g.cutter._get_heatmap_distance(x,y)
    return d

def dr(g):
    dis = d(g)
    dr = 5*(1/dis)
    return dr

def dd(g):
    print(f"D:   {d(g)}")
    print(f"DR:  {dr(g)}")
    print("")

g=GymEnv("data/frames/env_w_vics.h5", 30.0, -80.1, "resources/settings.json")


for i in range(5):
    dd(g)
    g.cutter.move('N')



