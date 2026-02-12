# Jonas Hallstrom, 10/16/2023, based on Freud documentation examples
# Updated 07/10/2024 to combine a few scripts into one

import freud
import numpy as np
import pickle
import glob

import gsd.hoomd
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
import pandas as pd

import matplotlib.pyplot as plt

""" Some initial setup """
dist_thresholds = [1.15]  # in units of the first RDF peak of the last frame
first_peak = int(glob.glob("FreudRDF_Peak*.png")[0].split("_Peak")[1].split(".pn")[0])

align_thresholds = [0, 10]  # in degree. 0 means alignment not considered in thresholding
symmetry_factor = 4  # for a square, 0, 90, 180, etc degrees are equivalent (mod 360/4)
mod_factor = int(360/symmetry_factor)

for dist_t in dist_thresholds:
    """ Find the edges """
    threshold =  dist_t*first_peak
    print("Threshold of", threshold, "for constant weighting")

    pos_files = glob.glob("../ExpPosNPY/*.npy")
    sorter = np.argsort([int(namestring.split("/")[-1].split("_")[0]) for namestring in pos_files])
    datafiles = [pos_files[num] for num in sorter]
    num_frames = len(datafiles)

    all_positions = []
    all_max = 0
    all_min = 0
    for i in range(num_frames):
        positions = np.load(datafiles[i])
        if (max:=np.max(positions)) > all_max:
            all_max = max
        if (min:=np.min(positions)) < all_min:
            all_min = min
    print(all_max, all_min, (all_max+all_min)/2)
    for i in range(num_frames):
        positions = np.load(datafiles[i])
        all_positions.append(positions - (all_max+all_min)/2)  # Center the positions 

    L = all_max - all_min
    box = freud.box.Box(L+5*threshold, L+5*threshold, 0)  # 1.5 because we DONT want PBC bonds for the experiment

    dt_all_nlists = []
    all_separations = []
    for j in range(num_frames):
        points = np.column_stack([all_positions[j], np.zeros(len(all_positions[j]))])
        aq = freud.locality.AABBQuery(box, points)
        query_result = aq.query(points, dict(r_max=threshold, exclude_ii=True))
        result = np.array(list(query_result))
        if len(result) == 0:
            print("Adding fake edge on frame {} where there are no edges".format(j))
            result = np.array([[1, 0, 0.99*threshold], [0, 1, 0.99*threshold]])
        nlist = result[:, :2]
        separations = result[:, 2]

        NR_nlist = []
        NR_sep = []
        for i in range(len(nlist)):
            if nlist[i][0] > nlist[i][1]:  # this removes any repeats
                NR_nlist.append(nlist[i])
                NR_sep.append(separations[i])

        nlist = np.array(NR_nlist)
        separations = np.array(NR_sep)

        dt_all_nlists.append(nlist)
        all_separations.append(separations)

    with open('PBCNR_nlists_dt{}.pkl'.format(int(threshold)), 'wb') as f:
        pickle.dump(dt_all_nlists, f)

    with open('PBCNR_separations_dt{}.pkl'.format(int(threshold)), 'wb') as f:
        pickle.dump(all_separations, f)


    for align_t in align_thresholds:
        folder = str(dist_t).replace(".", "p")+"_AT{}".format(align_t)
        """ Turn the edges into networks and calculate the curvatures """
        alignment_threshold = align_t
        print("Alignment threshold of {}".format(alignment_threshold))

        if align_t == 0:
            all_nlists = dt_all_nlists
        else:
            ang_files = glob.glob("../ExpAngNPY/*.npy")
            sorter = np.argsort([int(namestring.split("/")[-1].split("_")[0]) for namestring in ang_files])
            ang_files = [ang_files[num] for num in sorter]

            all_AT_nlists = []
            for i in range(num_frames):
                angles = np.load(ang_files[i])
                AT_nlist = []
                for pair in dt_all_nlists[i]:
                    a = int(pair[0])
                    b = int(pair[1])
                    # Though these are physical-distance neighbors, we will see if their NP orientations
                    #  are within our alignment threshold to be neighbors in the aligned-cluster or mesocrystal sense.
                    if angles[a] != -1 and angles[b] != -1:
                        if np.abs((angles[a]%mod_factor) - (angles[b]%mod_factor)) < alignment_threshold:
                            AT_nlist.append([a, b])
                    #else:
                    #    print(i)
                all_AT_nlists.append(np.array(AT_nlist))
            all_nlists = all_AT_nlists  # now our neighbor lists have been alignment thresholded too
        
        for i in range(len(all_nlists)):
            if len(all_nlists[i])==0:
                # make fake bond between the first two NPs if a frame has 0 bonds so as to not
                #  break all the future code with empty lists and such
                print("Adding fake bond to frame", str(i))
                all_nlists[i] = np.array([[0, 1]]) 

        all_dataframes = []
        for i in range(num_frames):
            nlist = all_nlists[i]
            nlist = [tuple(pair) for pair in nlist]

            weight_values = np.ones(len(all_nlists[i]))
            weights = dict(zip(nlist, weight_values))
            G = nx.from_edgelist(nlist)
            nx.set_edge_attributes(G, values = weights, name = 'weight')
            edges = np.array(G.edges)

            orc = OllivierRicci(G, alpha=0)
            orc.compute_ricci_curvature()
            G_orc = orc.G.copy()
            ricci_curvatures = list(nx.get_edge_attributes(G_orc, "ricciCurvature").values())

            frc = FormanRicci(G)
            frc.compute_ricci_curvature()
            G_frc = frc.G.copy()
            forman_curvatures = list(nx.get_edge_attributes(G_frc, "formanCurvature").values())

            df_dict = {"source":edges[:, 0],
                        "target":edges[:, 1],
                        "weight":weight_values,
                        "formanCurvature":forman_curvatures,
                        "ricciCurvature":ricci_curvatures}
            df = pd.DataFrame(df_dict)
            all_dataframes.append(df)

        with open(folder+'/PBCNR_curvature_dt{}at{}.pkl'.format(int(threshold), int(alignment_threshold)), 'wb') as f:
            pickle.dump(all_dataframes, f)


        """ Do some plotting """
        curv_data = all_dataframes
        orc_edges = [-0.91, -0.61, -0.41, -0.31, -0.01]  # additional edges at -infinity and +infinity assumed
        orc_edges = [-1*num for num in orc_edges] + orc_edges
        orc_edges = np.sort(orc_edges)
        orc_weights = [(orc_edges[i] + orc_edges[i+1])/2 for i in range(len(orc_edges)-1)]
        orc_weights = [-0.1+orc_edges[0]] + orc_weights + [0.1+orc_edges[-1]]

        total_orc = np.zeros(num_frames)
        binned_orc = np.zeros([num_frames, len(orc_edges)+1])
        total_frc = np.zeros(num_frames)
        num_bonds = np.zeros(num_frames)
        unique_orc = []
        for i in range(num_frames):
            orc = curv_data[i]["ricciCurvature"]
            for val in np.unique(orc):
                if (foo:=round(val, 3)) not in unique_orc:
                    unique_orc.append(foo)
            frc = curv_data[i]["formanCurvature"]

            total_orc[i] = np.sum(orc)
            total_frc[i] = np.sum(frc)
            num_bonds[i] = len(orc)

            for val in orc:
                if val>orc_edges[-1]:
                    binned_orc[i, -1] += 1
                else:
                    for j in range(len(orc_edges)):
                        if val<orc_edges[j]:
                            binned_orc[i, j] += 1
                            break

        #print(np.sort(unique_orc))

        min_tot_orc = np.argmin(total_orc)
        min_mean_orc = np.argmin(total_orc/num_bonds)

        plt.semilogy(binned_orc[:, 0], label="<{}".format(orc_edges[0]))
        for j in range(1, len(orc_edges)):
            plt.semilogy(binned_orc[:, j], label="[{},{})".format(orc_edges[j-1], orc_edges[j]))
        plt.semilogy(binned_orc[:, -1], label=">{}".format(orc_edges[-1]))
        plt.axvline(min_mean_orc, color="red", linestyle=":")
        plt.axvline(min_tot_orc, color="black", linestyle=":")
        plt.legend()
        plt.title("Binned ORC Counts")
        plt.savefig(folder+"/BinnedORC.png")
        plt.clf()
        plt.close()

        plt.plot(binned_orc[:, 0]*orc_weights[0], label="<{}".format(orc_edges[0]))
        for j in range(1, len(orc_edges)):
            plt.plot(binned_orc[:, j]*orc_weights[j], label="[{},{})".format(orc_edges[j-1], orc_edges[j]))
        plt.plot(binned_orc[:, -1]*orc_weights[-1], label=">{}".format(orc_edges[-1]))
        plt.axvline(min_mean_orc, color="red", linestyle=":")
        plt.axvline(min_tot_orc, color="black", linestyle=":")
        plt.title("Weighted Binned ORC Counts")
        plt.savefig(folder+"/WeightedBinnedORC.png")
        plt.clf()
        plt.close()

        plt.plot(binned_orc[:, 0]*orc_weights[0]/num_bonds, label="<{}".format(orc_edges[0]))
        for j in range(1, len(orc_edges)):
            plt.plot(binned_orc[:, j]*orc_weights[j]/num_bonds, label="[{},{})".format(orc_edges[j-1], orc_edges[j]))
        plt.plot(binned_orc[:, -1]*orc_weights[-1]/num_bonds, label=">{}".format(orc_edges[-1]))
        plt.axvline(min_mean_orc, color="red", linestyle=":")
        plt.axvline(min_tot_orc, color="black", linestyle=":")
        plt.title("Weighted Binned ORC Counts / Bonds")
        plt.savefig(folder+"/WeightedBinnedORCdivBonds.png")
        plt.clf()
        plt.close()

        plt.plot(num_bonds)
        plt.title("Number of Bonds")
        plt.savefig(folder+"/num_bonds.png")
        plt.clf()
        plt.close()

        plt.plot(total_orc)
        plt.axvline(min_mean_orc, color="red", linestyle=":")
        plt.axvline(min_tot_orc, color="black", linestyle=":")
        plt.title("Total Bond ORC. Min: {:.2f}".format(np.min(total_orc)))
        plt.savefig(folder+"/total_orc.png")
        plt.clf()
        plt.close()

        plt.plot(total_frc)
        plt.title("Total Bond FRC")
        plt.savefig(folder+"/total_frc.png")
        plt.clf()
        plt.close()

        plt.plot(total_orc / num_bonds)
        plt.axvline(min_mean_orc, color="red", linestyle=":")
        plt.axvline(min_tot_orc, color="black", linestyle=":")
        plt.title("Mean Bond ORC. Min: {:.2f}".format(np.min(total_orc / num_bonds)))
        plt.savefig(folder+"/mean_orc.png")
        plt.clf()
        plt.close()

        plt.plot(total_orc / num_bonds)
        #plt.axvline(min_mean_orc, color="red", linestyle=":")
        #plt.axvline(min_tot_orc, color="black", linestyle=":")
        #plt.title("Mean Bond ORC. Min: {:.2f}".format(np.min(total_orc / num_bonds)))
        plt.savefig(folder+"/mean_orc_notitle.png")
        plt.clf()
        plt.close()        

        plt.plot(total_frc / num_bonds)
        plt.title("Mean Bond FRC")
        plt.savefig(folder+"/mean_frc.png")
        plt.clf()
        plt.close()

        plt.plot(total_frc / num_bonds)
        #plt.title("Mean Bond FRC")
        plt.savefig(folder+"/mean_frc_notitle.png")
        plt.clf()
        plt.close()
