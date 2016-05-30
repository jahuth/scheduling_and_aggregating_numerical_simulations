

class XMLExperiment():
    def __init__(f='Experiments/new_test.xml'):
        import xml.etree.ElementTree as ET
        tree = ET.parse(f)
        root = tree.getroot()
        named_blocks = { b.attrib["name"] for b in root.iter("block") if "name" in b.attrib }
        unnamed_blocks = [ b in root.iter("block") if "name" not in b.attrib ]
        variabless = { b.attrib["name"] for b in root.iter("variables") if "name" in b.attrib }
        if root.find("title"):
            self.title = root.find("title").text
        else:
            self.title = ""
        if root.find("description"):
            self.description = root.find("description").text
        else:
            self.description = ""
        self.sessions = root.findall("session")