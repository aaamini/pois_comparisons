function input_cell = map_keys(input_cell, dict) 
% Implements a simple dictionary
    isInputCell = iscell(input_cell);
    if( ~isInputCell )
        input_cell = {input_cell};
    end

    for di = 1:length(input_cell)
        try % ignore undefined fields
            input_cell{di} = dict.(strrep(input_cell{di},'-','_')); % uses dynamic field referencing
        end
    end

    if( ~isInputCell )
        input_cell = input_cell{1};
    end

end