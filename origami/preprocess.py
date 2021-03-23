"""
preOrigami.py

This pre-processes the omnifold ntuples into input files for the Origami script.

Matt LeBlanc, CERN 2021
matt.leblanc@cern.ch
"""

import argparse
import h5py

import numpy as np
import matplotlib.pyplot as plt

import uproot3

import fastjet as fj

GEV = 1.e-3

def print_jets(jets):
    print("{0:>5s} {1:>10s} {2:>10s} {3:>10s}".format("jet #", "pt", "rap", "phi"))
    for ijet in range(len(jets)):
        print("{0:5d} {1:10.3f} {2:10.4f} {3:10.4f}".format(
            ijet, jets[ijet].pt(), jets[ijet].rap(), jets[ijet].phi()))


def select_event(pass190,
                 y_trackj1,
                 m_trackj1,
                 y_trackj2,
                 m_trackj2):

    pass_event = True

    if not pass190: 
        pass_event = False
        
    if np.abs(y_trackj1) > 2.1 :
        pass_event = False

    if np.abs(y_trackj2) > 2.1 :
        pass_event = False

    if m_trackj1==-99 :
        pass_event = False

    if m_trackj2==-99 :
        pass_event = False

    return pass_event


def get_tracks_in_leading_jet(pT_tracks, eta_tracks, phi_tracks, debug):
    assert(len(pT_tracks)==len(eta_tracks))
    assert(len(pT_tracks)==len(phi_tracks))

    if(debug): print("There are "+str(len(pT_tracks))+" tracks.")

    # set up our jet definition and a jet selector
    # R=1.0 jets, so stay within |rap|<1.5 for tracker acceptance
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 1.0)
    selector = fj.SelectorPtMin(5.0) & fj.SelectorAbsRapMax(1.5)

    # get the event
    event = []
    for iTrack in range(0,len(pT_tracks)):
        pj = fj.PseudoJet()

        pT  = float(pT_tracks[iTrack])
        eta = float(eta_tracks[iTrack]) # sorry Dag
        phi = float(phi_tracks[iTrack])
        m   = float(0.13957)

        pj.reset_PtYPhiM(pT, eta, phi, m)

        event.append( pj );

    # cluster it
    jets = selector(jet_def(event))

    # print out some information about the event and clustering
    if(debug):
        print("Event has {0} particles".format(len(event)))
        print("jet definition is:",jet_def)
        print("jet selector is:", selector,"\n")

        # print the jets
        print_jets(jets)

    tracks=[]
    if(len(jets)>0):
        if(jets[0].has_constituents()):
            if(debug): print("Leading jet has "+str(len(jets[0].constituents()))+" constituents")
            tracks = jets[0].constituents()

        return tracks,jets[0]
    else: return tracks,None

def calc_jss(jet, observable, radii,  debug):

    jss=[]

    for R in radii:
        jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
        selector = fj.SelectorPtMin(5.0) & fj.SelectorAbsRapMax(1.5)
        jets = selector( jet_def( jet.constituents() ) )
        if(len(jets)>0):
            jet = jets[0]
            if observable=='rho' :
                jss.append( np.log( jet.m()*jet.m()/( jet.perp()*jet.perp() ) ) )
            if observable=='ljp_zdr' : 
                ljp_tuple_list = []
                
                jet_def_ca = fj.JetDefinition(fj.cambridge_algorithm, fj.JetDefinition.max_allowable_R)
                rc = fj.Recluster(jet_def_ca)
                jet = rc.result(jets[0])
                
                j1 = fj.PseudoJet()
                j2 = fj.PseudoJet()
                jj = jet
                while jj.has_parents(j1,j2):
                    if(j2.pt2() > j1.pt2()):
                        print("oh no oh no")
                    
                    z  = j2.pt2() / ( j1.pt2()+j2.pt2() ) 
                    dr = j1.delta_R(j2)

                    ljp_tuple_list.append([ np.log(R/dr), np.log(1/z) ])

                    # follow harder branch
                    jj = j1;
                    
                #print(np.array(ljp_tuple_list).shape)
                jss.append( ljp_tuple_list )

        else:
            if( observable=='ljp_zdr' ): jss.append([-100,-100])
            else :                       jss.append(-100)
            
    return jss

"""
def calc_sd_jss(jet, observable, betas, zcuts, debug):

    jss=[]

    for beta in betas:
        for zcut in zcuts:
            R=fj.JetDefinition.max_allowable_R
            jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
            selector = fj.SelectorPtMin(5.0) & fj.SelectorAbsRapMax(1.5)
            jets = selector( jet_def( jet.constituents() ) )
            sd = fj.contrib.softdrop(beta,zcut)
            jets=sd(jets)
            if(len(jets)>0):
                jet = jets[0]
                if observable=='rho' :
                    jss.append( np.log( jet.m()*jet.m()/( jet.perp()*jet.perp() ) ) )
            else:
                jss.append(-1)

    return jss
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--inFile", dest='inFile', default="", required=True, help="Input file.")
    parser.add_argument("--inTree", dest='inTree', default="TinyJetTrees_R32", help="Input tree.")
    parser.add_argument("--outDir", dest='outDir', default="", help="Output directory.")
    parser.add_argument("--isData", dest='isData', action='store_true', help="Is this data?")
    parser.add_argument("--startEvent", dest='startEvent', type=int, default=-1, help="event to start processing from")
    parser.add_argument("--stopEvent", dest='stopEvent', type=int, default=-1, help="event to stop processing at")
    parser.add_argument("--label", dest='label', default="TEST", help="Label for output")
    args = parser.parse_args()

    print("Input file is:\n"+str(args.inFile))

    print("Writing output events to ",args.inFile.split("/")[-1])
    out_file_str = args.outDir+args.label+"_"+args.inFile.split("/")[-1]+".h5"
    
    tree = uproot3.open(str(args.inFile))[args.inTree]
    print("There are "+str(len(tree))+" events")

    # this is the array of the tracks in the leading jet of each event, at reco- and truth-level
    # it is written out as an .h5, as it is  needed downstream for EMD calculations
    track_array = np.zeros( (len(tree), 100, 6 ) )

    # substructure variables for plotting
    reco_rhos = []
    #reco_sd_rhos = []

    reco_ljps = []

    start_at = args.startEvent
    stop_at = args.stopEvent
    if(args.startEvent<0): start_at=0
    if(args.stopEvent<0): stop_at=len(tree)

    branches = [
        "weight_mc",
        "pT_tracks",
        "truth_pT_tracks",
        "eta_tracks",
        "truth_eta_tracks",
        "phi_tracks",
        "truth_phi_tracks",
        "pass190",
        "truth_pass190",
        "y_trackj1",
        "truth_y_trackj1",
        "y_trackj2",
        "truth_y_trackj2",
        "m_trackj1",
        "truth_m_trackj1",
        "m_trackj2",
        "truth_m_trackj2"
    ]

    for weight_mc,pT_tracks,truth_pT_tracks,y_tracks,truth_y_tracks,phi_tracks,truth_phi_tracks,pass190,truth_pass190,y_trackj1,truth_y_trackj1,y_trackj2,truth_y_trackj2,m_trackj1,truth_m_trackj1,m_trackj2,truth_m_trackj2 in tree.iterate(
            branches,
            outputtype=tuple, entrysteps=np.inf, namedecode='utf-8',
            entrystart=start_at,
            entrystop=stop_at,
    ):
        for idx in range(0, stop_at):
            debug=False
            if((idx==0) or (idx%1000==0)): debug=True

            if(debug): print("preOrigami\tevent:"+str(idx)+"\tweight_mc: "+str(weight_mc[idx]))

            pass_reco = select_event(pass190[idx],
                                     y_trackj2[idx],
                                     m_trackj1[idx],
                                     y_trackj2[idx],
                                     m_trackj2[idx])

            pass_true = select_event(truth_pass190[idx], 
                                     truth_y_trackj2[idx],
                                     truth_m_trackj1[idx],
                                     truth_y_trackj2[idx],
                                     truth_m_trackj2[idx])

            if not (pass_reco or pass_true):
                if debug :
                    if not pass_reco :
                        print("Fails reco selection!")
                    if not pass_true :
                        print("Fails truth selection!")
                continue
            
            reco_tracks = []
            truth_tracks = []

            reco_mass = []

            if(pass_reco):
                reco_tracks,reco_jet = get_tracks_in_leading_jet(pT_tracks[idx],
                                                                 y_tracks[idx],
                                                                 phi_tracks[idx],
                                                                 debug)
                if(len(reco_tracks)>0):
                    for iTrack in range(0,len(reco_tracks)) :
                        track_array[idx,iTrack,0] = reco_tracks[iTrack].pt()
                        track_array[idx,iTrack,1] = reco_tracks[iTrack].rapidity()
                        track_array[idx,iTrack,2] = reco_tracks[iTrack].phi()
                        
                    reco_rhos.append( calc_jss(reco_jet,
                                               observable='rho',
                                               radii=[1.0,0.8,0.6,0.4,0.2],
                                               debug=debug) )

                    reco_ljps.append( calc_jss(reco_jet,
                                               observable='ljp_zdr',
                                               radii=[1.0],
                                               debug=debug) )                    

                    '''
                    reco_sd_rhos.append( calc_sd_jss(reco_jet,
                                                     observable='sd_rho',
                                                     betas=[2.0,1.8,1.6,1.4,1.2,1.0,0.8,0.6,0.4,0.2,0.0],
                                                     zcuts=[0.1],
                                                     debug=True) )
                    '''

            if(pass_true):
                truth_tracks,truth_jet = get_tracks_in_leading_jet(truth_pT_tracks[idx],
                                                                   truth_y_tracks[idx],
                                                                   truth_phi_tracks[idx],
                                                                   debug)
                
                if(len(truth_tracks)>0):
                    for iTrack in range(0,len(truth_tracks)) :
                        track_array[idx,iTrack,3] = truth_tracks[iTrack].pt()
                        track_array[idx,iTrack,4] = truth_tracks[iTrack].rapidity()
                        track_array[idx,iTrack,5] = truth_tracks[iTrack].phi()

    if(debug): print(track_array[idx])
    
print("Plotting substructure observables ...")

#for idx,R in enumerate([1.0, 0.8, 0.6, 0.4, 0.2]):

print(np.array(reco_rhos, dtype=object).shape)

# ungroomed mass
plt.hist([np.array(reco_rhos)[:,0],
          np.array(reco_rhos)[:,1],
          np.array(reco_rhos)[:,2],
          np.array(reco_rhos)[:,3],
          np.array(reco_rhos)[:,4]], 
         bins=50,
         range=(-10.,0.0),
         histtype='step',
         label=['R=1.0','R=0.8','R=0.6','R=0.4','R=0.2'])

plt.legend(loc='center right', frameon=False, fontsize="x-large")
plt.xlabel('rho', fontsize="x-large");
plt.show()
plt.savefig("out_origami/reco_rhos.png")
plt.close()

# lund jet plane

plt.hist2d(np.array(reco_ljps[:,0], dtype=object), 
           np.array(reco_ljps[:,1], dtype=object),
           bins=[50,50],
           range=((0.0, 6.0), (0.0, 6.0)),
           cmap=plt.get_cmap('inferno'))

#plt.legend(loc='center right', frameon=False, fontsize="x-large")
plt.xlabel('ln(R/DeltaR)', fontsize="x-large");
plt.ylabel('ln(1/z)', fontsize="x-large");
plt.show()
plt.savefig("out_origami/ljp.png")
plt.close()

print("Writing output.")

#h5f = h5py.File(out_file_str, 'w')
#h5f.create_dataset('track_array', data=track_array)
#h5f.close()

print("Closing output.")



print("All done!")
