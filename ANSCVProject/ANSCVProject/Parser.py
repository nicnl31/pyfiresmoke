class Point:
    '''
    DESCRIPTION: alias for coordinates. All attributes are publicly
    accessible. No need for setters/getters.
    '''
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return f"{self.x},{self.y}"


class Object:
    '''
    DESCRIPTION: is the object denoted by the region of interest. All
    attributes are publicly accessible. No need for setters/getters
    '''
    def __init__(self, id: int, name: str, minp: Point, maxp: Point, counter: int):
        self.id: int = id
        self.name: str = name
        self.top_left: Point = minp
        self.bottom_right: Point = maxp
        self.counter: int = counter

    def __str__(self):
        return f"{self.id},{self.name},{self.top_left},{self.bottom_right},{self.counter}"


def readRegionsOfInterest(path) -> list:
    '''
    DESCRIPTION: takes in a `path`, reads the file and returns a list
    of ROI objects (class `Object`).
    '''
    f = open(path, "r")
    contents = f.read()
    
    lines = contents.split("\n")

    results = []
    i = 0

    while i < len(lines):
        if i == 0:
            i += 1
            continue
        else:
            EXACT_ARG_COUNT = 7
            line = lines[i]
            params = line.split(",")
            if len(params) != EXACT_ARG_COUNT:
                i += 1
                continue
            
            results.append(Object(int(params[0]), params[1],
                                  Point(float(params[2]), float(params[3])),
                                  Point(float(params[4]), float(params[5])),
                                  int(params[6]))
                           )
        i += 1
    f.close()
    
    return results

def main():
    # NOTE: replace the path with either the absolute path or the relative path
    #   of the CSV file.
    for roi in readRegionsOfInterest("results.csv"):
        print(roi)

if __name__ == "__main__":
    main()

            
