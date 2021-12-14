import awkward
import uproot
# import uproot_methods

import time

class Event_table:
    def __init__(self, *args):
        self.events = awkward.Table()
        for i in range(len(args)):
            self.events[f"{i}"] = args[i]
    def __repr__(self):
        return "<Event Table>"
    def rename_columns(self, column_names):
        cols = self.events.columns
        for i, col in enumerate(cols):
            self.events[column_names[i]] = self.events[col]
            del self.events[col]
    def extend(self, toadd):
        assert (self.__repr__() == toadd.__repr__()),"Trying to combine different types of events"
        assert (self.events.columns == toadd.events.columns),"Trying to combine events with different column names" 
        self.events = awkward.concatenate([self.events, toadd.events])
    def size(self):
        return len(self.events[self.events.columns[0]])
    def make_4vector_ptetaphim(self, ptcol=None, etacol=None, phicol=None, mcol=None, newcol=None):
        if mcol==None:
            self.events[newcol] = uproot_methods.TLorentzVectorArray.from_ptetaphim(self.events[ptcol],
                                                                                    self.events[etacol],
                                                                                    self.events[phicol],
                                                                                    self.events[ptcol]*0.0)
        else:
            self.events[newcol] = uproot_methods.TLorentzVectorArray.from_ptetaphim(self.events[ptcol],
                                                                                    self.events[etacol],
                                                                                    self.events[phicol],
                                                                                    self.events[mcol])
    def force_allcolumns_jagged(self):
        for col in self.events.columns:
            self.events[col] = awkward.fromiter(self.events[col])
    def save(self, filename, mode='w'):
        awkward.save(filename+".awkd", self.events, mode=mode)
    def load(self, filename):
        self.events = awkward.load(filename)
#     def prune(self):
#         self.events = self.events[self.events.selection==True]
#         self.events = self.events.deepcopy()

def prepare_dataset_table(filenames, treename, branch_list, col_names, entrysteps=10000, outputype=Event_table):
    t = time.time()
    count, goodEvents = 0, None
    for Events in uproot.iterate(filenames, treename, branch_list,namedecode="utf-8", entrysteps=entrysteps, 
                                 outputtype=outputype):
        Events.rename_columns(col_names)
        count = count + Events.size()
        if goodEvents != None:
            goodEvents.extend(Events)
        else:
            goodEvents = Events
    print("Total number of events: ",count)
    print("Events in returned Object: ",goodEvents.size())
    print("Loading the data took ", time.time()-t, " seconds")
    return goodEvents
