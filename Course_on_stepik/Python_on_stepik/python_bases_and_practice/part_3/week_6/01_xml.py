from xml.etree import ElementTree

tree = ElementTree.parse('01_example.xml')
root = tree.getroot()
# use root = ElementTree.fromstring(string_xml_data) to parse from string

print(root)
print(root.tag, root.attrib)

for child in root:
    print(child.tag, child.attrib)

print(root[0][0].text)

for element in root.iter('scores'):
    score_sum = 0
    for child in element:
        score_sum += float(child.text)
    print(score_sum)