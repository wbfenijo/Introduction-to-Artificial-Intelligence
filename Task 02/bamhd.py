import json
import sys
from queue import PriorityQueue
from math import radians, cos, sin, asin, sqrt

class BaMHD(object):
    def __init__(self, db_file=r'C:\Users\adamy\OneDrive\Plocha\ItAI\Task 02\ba_mhd_db.json'):
        # Initialize BaMHD object, load data from json file
        self.data = json.load(open(db_file, 'r'))

    def distance(self, stop1, stop2):
        # Return distance between two stops in km.
        if isinstance(stop1, BusStop): stop1 = stop1.name
        if isinstance(stop2, BusStop): stop2 = stop2.name
        coords1 = self.data['bus_stops'][stop1]
        coords2 = self.data['bus_stops'][stop2]

        def haversine(lon1, lat1, lon2, lat2):
            # (You don`t need to understand following code - it`s just geo-stuff)
            # Calculate the great circle distance between two points on the earth (specified in
            # decimal degrees)
            # Courtesy of http://stackoverflow.com/a/15737218

            # convert decimal degrees to radians
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            # haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            km = 6367 * c
            return km

        return haversine(coords1[0], coords1[1], coords2[0], coords2[1])

    def neighbors(self, stop):
        # Return neighbors for a given stop
        return self.data['neighbors'][stop.name if isinstance(stop, BusStop) else stop]

    def stops(self):
        # Return list of all stops (names only)
        return self.data['neighbors'].keys()


class BusStop(object):
    # Object representing node in graph traversal. Includes name, parent node, and total cost of
    # path from root to this node (i.e. distance from start).
    def __init__(self, name, parent = None, pathLength = 0):
        self.name = name
        self.parent = parent
        self.pathLength = pathLength

    def traceBackPath(self):
        # Returns path represented by this node as list of node names (bus stop names).
        if self.parent == None:
            return [self.name]
        else:
            path = self.parent.traceBackPath()
            path.append(self.name)
            return path


def findPathUniformCost(bamhd, stopA, stopB):
    # Implement Uniform-cost search to find shortest path between two MHD stops in Bratislava.
    # Return a list of MHD stops, print how many bus stops were added to the "OPEN list"
    # and total path length in km.

    ### Your code here ###

    priority_queue = PriorityQueue()
    start = BusStop(stopA)
    stop = BusStop(stopB)
    priority_queue.put((start.pathLength, start))

    visited = {
        start.name: start
    }

    visitedName = [start.name]

    while priority_queue:
        
        current_cost, currentStop = priority_queue.get()

        if currentStop.name == stop.name:
            path = currentStop.traceBackPath()
            print('\t{} bus stops in "OPEN list", length = {}km'.format(len(path), currentStop.pathLength))
            return path
        
        for neighbor in bamhd.neighbors(currentStop.name):

            neighborStop = BusStop(neighbor,currentStop, current_cost)
            distance = bamhd.distance(currentStop,neighborStop)
            neighborStop.pathLength += distance
            if neighbor not in visitedName or neighborStop.pathLength < visited[neighbor].pathLength:
                visited[neighbor] = neighborStop
                visitedName.append(neighbor)
                priority_queue.put((neighborStop.pathLength, neighborStop))

    



def findPathAStar(bamhd, stopA, stopB):
    # Implement A* search to find shortest path between two MHD stops in Bratislava.
    # Return a list of MHD stops, print how many bus stops were added to the "OPEN list"
    # and total path length in km.

    ### Your code here ###
    print('\t{} bus stops in "OPEN list", length = {}km'.format(123, 42))
    return []


if __name__ == "__main__":
    # Initialization
    bamhd = BaMHD() #slovnik slovnikov, najprv neighbors alebo busstops a potom meno zastavky a vnutri list susedov
    # Examples of function usage:
    # -> accessing the list of bus stops (is 'Zoo' a bus stop?)
    print('Zoo' in bamhd.stops())
    # -> get neighbouring bus stops. Parameters can be string or BusStop object
    print(bamhd.neighbors('Zochova'))
    # -> get distance between two bus stops (in km). Parameters can be string or BusStop objects
    print(bamhd.distance('Zochova', 'Zoo'))
    # -> get whole path from last node of search algorithm
    s1 = BusStop('Zoo')     # some dummy data
    s2 = BusStop('Lafranconi', s1)
    s3 = BusStop('Park kultury', s2)
    print(s3.traceBackPath())
    # -> using priority queue
    pq = PriorityQueue()
    pq.put((3, 'Not important stuff'))  # pq.put((priority, object))
    pq.put((1, 'Important stuff'))
    pq.put((2, 'Medium stuff'))
    print(pq.get()[1])
    print(pq.get()[1])
    print(pq.get()[1])
    print(pq.empty())


    # Your task: find best route between two stops with:
    # A) Uniform-cost search
    print('Uniform-cost search:')
    print('Zoo - Aupark:')
    path = findPathUniformCost(bamhd, 'Zoo', 'Aupark')
    print('\tpath: {}'.format(path))

    print('VW - Astronomicka:')
    path = findPathUniformCost(bamhd, 'Volkswagen', 'Astronomicka')
    print('\tpath: {}'.format(path))

    # B) A* search
    print('\nA* search:')
    print('Zoo - Aupark:')
    path = findPathAStar(bamhd, 'Zoo', 'Aupark')
    print('\tpath: {}'.format(path))

    print('VW - Astronomicka:')
    path = findPathAStar(bamhd, 'Volkswagen', 'Astronomicka')
    print('\tpath: {}'.format(path))
