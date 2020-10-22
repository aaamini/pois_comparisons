function dataCell = map_dataset(dataCell)
% Maps dataset names

data_map = struct();
data_map.toxic_breast = 'Breast Cancer Toxicity';
data_map.toxic_all_new = 'Colorectal Cancer Toxicity';
data_map.amazon = 'Amazon Reviews';
data_map.movie = 'Movie Reviews';
data_map.PROall = 'Colorectal Cancer PRO Data';
data_map.PRO_breast_all = 'Breast Cancer PRO Data';

dataCell = vis.map_keys(dataCell, data_map);
