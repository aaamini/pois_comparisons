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
    % for di = 1:length(datasetArray)
    %     dataset = datasetArray{di};
    %     switch dataset
    %         case 'highly_cor3'
    %             dataset = 'Highly-correlated (Simulated)'
    %         case 'toxic_all_new'
    %             dataset = 'Toxicity (All)';
    %         case 'PROall'
    %             dataset = 'PRO (All)';
    %         case '20news-with-outliers'
    %             dataset = '20News w/ Outliers';
    %         case '20news'
    %             dataset = '20News';
    %         case 'crash-severity'
    %             dataset = 'Crash Severity';
    %         case 'crime-lapd'
    %             dataset = 'Crime LAPD';
    %         case 'brca'
    %             dataset = 'BRCA';
    %         case 'classic3'
    %             dataset = 'Classic3';
    %         case 'toxic_5FUplusOxa'
    %             dataset = 'toxic 5FU+Oxa';
    %         case 'toxic_CapeplusOxa'
    %             dataset = 'toxic Cape+Oxa'
    %         case 'toxic_5FU'
    %             dataset = 'toxic 5FU'
    %         case 'toxic_Cape'
    %             dataset = 'toxic Cape'
    %         case 'PRO_CapeplusOxa'    
    %             dataset = 'PRO Cape+Oxa'
    %         case 'PRO_5FUplusOxa'
    %             dataset = 'PRO 5FU+OXa'
    %         case 'PRO_5FU'
    %             dataset = 'PRO 5FU'
    %         case 'PRO_Cape'
    %             dataset = 'PRO Cape'
    %         case 'books'
    %             dataset = 'Book Ratings';
    %         case 'movie'
    %             dataset = 'Movie Ratings';
    %         case 'amazon'
    %             dataset = 'Amazon Reviews';
    %         otherwise
    %             dataset = datasetArray{di}; % Default keep label
                
    %     end
    %     datasetArray{di} = dataset;
    % end
    % if( ~isInputCell )
    %     datasetArray = datasetArray{1};
    % end
end