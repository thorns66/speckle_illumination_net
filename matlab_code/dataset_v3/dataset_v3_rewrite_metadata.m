function [value,changed] = dataset_v3_rewrite_metadata(value,oldId,newId,newSplit,oldRoots,newRoot,field)
% Only text metadata is changed; numeric/logical arrays are never assigned.
if nargin<7, field=''; end
changed=false;
if isstruct(value)
    names=fieldnames(value);
    for k=1:numel(value)
        for j=1:numel(names)
            name=names{j};
            [v,c]=dataset_v3_rewrite_metadata(value(k).(name),oldId,newId,newSplit,oldRoots,newRoot,name);
            if c, value(k).(name)=v; changed=true; end
        end
    end
elseif iscell(value)
    for k=1:numel(value)
        [v,c]=dataset_v3_rewrite_metadata(value{k},oldId,newId,newSplit,oldRoots,newRoot,field);
        if c, value{k}=v; changed=true; end
    end
elseif ischar(value)&&isrow(value)
    previous=value;
    if strcmp(field,'split'), value=newSplit; end
    if strcmp(value,oldId), value=newId; end
    for k=1:numel(oldRoots)
        old=oldRoots{k};
        if strcmp(value,old)||startsWith(value,[old filesep])
            suffix=value(numel(old)+1:end);
            if strcmp(suffix,[filesep oldId])||startsWith(suffix,[filesep oldId filesep])
                suffix=[filesep newId suffix(numel(oldId)+2:end)];
            end
            value=[newRoot suffix]; break;
        end
    end
    changed=~isequal(value,previous);
elseif isstring(value)
    for k=1:numel(value)
        [v,c]=dataset_v3_rewrite_metadata(char(value(k)),oldId,newId,newSplit,oldRoots,newRoot,field);
        if c, value(k)=string(v); changed=true; end
    end
end
end
