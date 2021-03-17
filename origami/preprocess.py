"""
preOrigami.py

This pre-processes the omnifold ntuples into input files for the Origami script.

Matt LeBlanc, CERN 2021
matt.leblanc@cern.ch
"""

import argparse

import numpy as np
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
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.6)
    selector = fj.SelectorPtMin(5.0) & fj.SelectorAbsRapMax(2.1)

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
    
    #if(debug): print_jets(tracks)

    return tracks


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

    #print("Writing output events to ",args.inFile.split("/")[-1])
    #out_file_str = args.outDir+args.label+"_"
    
    tree = uproot3.open(str(args.inFile))[args.inTree]
    print("There are "+str(len(tree))+" events")

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

        for idx in range(0,len(tree)):
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

            if(pass_reco):
                reco_tracks = get_tracks_in_leading_jet(pT_tracks[idx],
                                                        y_tracks[idx],
                                                        phi_tracks[idx],
                                                        debug)

            if(pass_true):
                truth_tracks = get_tracks_in_leading_jet(truth_pT_tracks[idx],
                                                         truth_y_tracks[idx],
                                                         truth_phi_tracks[idx],
                                                         debug)

    print("Writing output.")
    print("Closing output.")

print("All done!")
