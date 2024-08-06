import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import imageio.v2 as imageio
from scipy.stats import linregress


#------------------------------
# ANIMATION PLOT
#------------------------------
def animate(networks, pos_type="circular", path="output.gif", plot_every=1):
    temp_files = []

    for i, G in enumerate(networks):
        # skip if greater than 1
        if i % plot_every != 0:
            continue

        # get positions based on type
        if pos_type == "circular":
            pos = nx.circular_layout(G)
        elif pos_type == "random":
            pos = nx.random_layout(G)
        elif pos_type == "spring":
            pos = nx.spring_layout(G)
        else:
            raise ValueError("Invalid pos_type. Use 'circular', 'random', or 'spring'.")

        # initialize fig & plot
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)

        # min & max pos for box
        tmpx = [pos[n][0] for n in pos]
        tmpy = [pos[n][1] for n in pos]
        Lxmin, Lxmax = min(tmpx) - 0.2, max(tmpx) + 0.2
        Lymin, Lymax = min(tmpy) - 0.2, max(tmpy) + 0.2

        # draw box
        ax.axhline(y=Lymin)
        ax.axvline(x=Lxmin)
        ax.axhline(y=Lymax)
        ax.axvline(x=Lxmax)

        # set node size & color
        if nx.is_directed(G):
            node_sizes = [G.in_degree(n) * 100 for n in G.nodes()]
            node_colors = [G.out_degree(n) for n in G.nodes()]
        else:
            node_sizes = [G.degree(n) * 100 for n in G.nodes()]
            node_colors = [G.degree(n) for n in G.nodes()]

        # draw network
        nx.draw(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, with_labels=True, cmap=plt.cm.viridis)
        
        # save plot to PNG file
        temp_file = f"tmp-{i + 1}.png"
        temp_files.append(temp_file)
        plt.savefig(temp_file)
        plt.close()

    # knit PNGs into a GIF
    images = [imageio.imread(filename) for filename in temp_files]
    imageio.mimsave(path, images, duration=10 / len(temp_files))

    # clean up temp files
    for filename in temp_files:
        os.remove(filename)

    # show GIF
    from IPython.display import display, Image
    display(Image(data=open(path, 'rb').read(), format='png', width=800))

#------------------------------
# NETWORK CENTRALITY CORRELATION PLOTS
#------------------------------
def plot_centrality_correlation(G, path=""):
    if nx.is_directed(G):
        centrality_measures = {
            "In Degree Centrality": nx.in_degree_centrality(G),
            "Out Degree Centrality": nx.out_degree_centrality(G),
            "In Closeness Centrality": nx.closeness_centrality(G.reverse(copy=True), wf_improved=False),
            "Out Closeness Centrality": nx.closeness_centrality(G, wf_improved=False),
            "Betweenness Centrality": nx.betweenness_centrality(G.to_undirected())
        }
    else:
        centrality_measures = {
            "Degree Centrality": nx.degree_centrality(G),
            "Closeness Centrality": nx.closeness_centrality(G),
            "Betweenness Centrality": nx.betweenness_centrality(G)
        }

    df = pd.DataFrame(centrality_measures)

    sns.pairplot(df)
    plt.suptitle("Centrality Correlation Plots", y=1.02)

    if path:
        plt.savefig(path)
    plt.show()

#------------------------------
# AVERAGE DEGREE
#------------------------------
def ave_degree(G):
    if nx.is_directed(G):
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        print("Average In-Degree: ", np.mean(list(in_degrees.values())))
        print("Average Out-Degree: ", np.mean(list(out_degrees.values())))
    else:
        degrees = dict(G.degree())
        print("Average Degree: ", np.mean(list(degrees.values())))


#------------------------------
# PLOT DEGREE DISTRIBUTION
#------------------------------
def plot_degree_distribution(G,type="in",path="",fit=False):
    if not nx.is_directed(G):
        data = G.degree()
        type = ""
    else:
        if type == "in":
            data = G.in_degree()
        elif type == "out":
            data = G.out_degree()
        else:
            raise ValueError("Invalid type for directed graph. Use 'in' or 'out'.")

    degrees = np.array([d for n, d in data])
    hist, bins = np.histogram(degrees, bins=30, density=True)
    cdf = np.cumsum(hist) * np.diff(bins)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # plot PDF
    ax[0].loglog(bins[:-1], hist, 'bo', markersize=8)
    ax[0].set_title(f'{type.capitalize()} Degree Distribution (PDF)')
    ax[0].set_xlabel('Degree')
    ax[0].set_ylabel('Frequency')

    # plot CDF
    ax[1].loglog(bins[:-1], 1 - cdf, 'bo', markersize=8)
    ax[1].set_title(f'{type.capitalize()} Degree Distribution (CDF)')
    ax[1].set_xlabel('Degree')
    ax[1].set_ylabel('Complementary Cumulative Frequency')

    # fit power-law
    if fit:
        def fit_power_law(x, y):
            log_x = np.log(x)
            log_y = np.log(y)
            slope, intercept, _, _, _ = linregress(log_x, log_y)
            return slope, intercept

        # fit PDF
        slope_pdf, intercept_pdf = fit_power_law(bins[:-1], hist)
        ax[0].loglog(bins[:-1], np.exp(intercept_pdf) * bins[:-1] ** slope_pdf, 'r-', label=f'Fit: alpha={-slope_pdf:.2f}')
        ax[0].legend()

        # fit CDF
        slope_cdf, intercept_cdf = fit_power_law(bins[:-1], 1 - cdf)
        ax[1].loglog(bins[:-1], np.exp(intercept_cdf) * bins[:-1] ** slope_cdf, 'r-', label=f'Fit: alpha={-slope_cdf:.2f}')
        ax[1].legend()

    # save/show plot
    if path:
        plt.savefig(path)
    plt.show()

#------------------------------
# NETWORK PLOTTING FUNCTION
#------------------------------
def plot_network(G,node_color="degree",layout="random"):
    
    # POSITIONS LAYOUT
    N=len(G.nodes)
    if(layout=="spring"):
        # pos=nx.spring_layout(G,k=50*1./np.sqrt(N),iterations=100)
        pos=nx.spring_layout(G)

    if(layout=="random"):
        pos=nx.random_layout(G)

    #INITALIZE PLOT
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)

    # NODE COLORS
    cmap=plt.cm.get_cmap('Greens')

    # DEGREE 
    if node_color=="degree":
            centrality=list(dict(nx.degree(G)).values())
  
    # BETWENNESS 
    if node_color=="betweeness":
            centrality=list(dict(nx.betweenness_centrality(G)).values())
  
    # CLOSENESS
    if node_color=="closeness":
            centrality=list(dict(nx.closeness_centrality(G)).values())

    # NODE SIZE CAN COLOR
    node_colors = [cmap(u/(0.01+max(centrality))) for u in centrality]
    node_sizes = [4000*u/(0.01+max(centrality)) for u in centrality]

    # #PLOT NETWORK
    nx.draw(G,
            with_labels=True,
            edgecolors="black",
            node_color=node_colors,
            node_size=node_sizes,
            font_color='white',
            font_size=18,
            pos=pos
            )

    plt.show()

#------------------------------
# NETWORK SUMMARY FUNCTION
#------------------------------
def network_summary(G):

    def centrality_stats(x):
        x1=dict(x)
        x2=np.array(list(x1.values())); #print(x2)
        print("	min:" ,min(x2))
        print("	mean:" ,np.mean(x2))
        print("	median:" ,np.median(x2))
        # print("	mode:" ,stats.mode(x2)[0][0])
        print("	max:" ,max(x2))
        x=dict(x)
        sort_dict=dict(sorted(x1.items(), key=lambda item: item[1],reverse=True))
        print("	top nodes:",list(sort_dict)[0:6])
        print("	          ",list(sort_dict.values())[0:6])

    try: 
        print("GENERAL")
        print("	number of nodes:",len(list(G.nodes)))
        print("	number of edges:",len(list(G.edges)))

        print("	is_directed:", nx.is_directed(G))
        print("	is_weighted:" ,nx.is_weighted(G))


        if(nx.is_directed(G)):
            print("IN-DEGREE (NORMALIZED)")
            centrality_stats(nx.in_degree_centrality(G))
            print("OUT-DEGREE (NORMALIZED)")
            centrality_stats(nx.out_degree_centrality(G))
        else:
            print("	number_connected_components", nx.number_connected_components(G))
            print("	number of triangle: ",len(nx.triangles(G).keys()))
            print("	density:" ,nx.density(G))
            print("	average_clustering coefficient: ", nx.average_clustering(G))
            print("	degree_assortativity_coefficient: ", nx.degree_assortativity_coefficient(G))
            print("	is_tree:" ,nx.is_tree(G))

            if(nx.is_connected(G)):
                print("	diameter:" ,nx.diameter(G))
                print("	radius:" ,nx.radius(G))
                print("	average_shortest_path_length: ", nx.average_shortest_path_length(G))

            #CENTRALITY 
            print("DEGREE (NORMALIZED)")
            centrality_stats(nx.degree_centrality(G))

            print("CLOSENESS CENTRALITY")
            centrality_stats(nx.closeness_centrality(G))

            print("BETWEEN CENTRALITY")
            centrality_stats(nx.betweenness_centrality(G))
    except:
        print("unable to run")

#------------------------------
# ISOLATE GCC
#------------------------------
def isolate_GCC(G):
    comps = sorted(nx.connected_components (G),key=len, reverse=True) 
    nodes_in_giant_comp = comps[0]
    return nx.subgraph(G, nodes_in_giant_comp)