function result = dataset_v3_numeric_equal(a,b)
% Compare every numerical payload after MAT serialization, including seeds.
result=strcmp(class(a),class(b)); if ~result, return; end
if isstruct(a)
    names=fieldnames(a); result=isequal(size(a),size(b))&&isequal(names,fieldnames(b));
    if ~result, return; end
    for k=1:numel(a)
        for j=1:numel(names)
            if ~dataset_v3_numeric_equal(a(k).(names{j}),b(k).(names{j})), result=false; return; end
        end
    end
elseif iscell(a)
    result=isequal(size(a),size(b)); if ~result, return; end
    for k=1:numel(a)
        if ~dataset_v3_numeric_equal(a{k},b{k}), result=false; return; end
    end
elseif isnumeric(a)||islogical(a)
    result=isequaln(a,b);
elseif ischar(a)||isstring(a)
    result=true; % Text changes are the sole purpose of this migration.
else
    result=isequaln(a,b);
end
end
