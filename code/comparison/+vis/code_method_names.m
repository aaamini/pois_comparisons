function out = code_method_names(batch)

mcode_list = {};
for mi = 1:length(batch)
    mname = batch{mi};
    mcode = [];
    for s = strsplit(mname,'-')
        mcode = [mcode upper(s{:}(1))];
    end
    mcode_list = [mcode_list, mcode];
    
end
out = strjoin(mcode_list,'-');