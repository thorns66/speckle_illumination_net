function cell_write_json(path, value)
%CELL_WRITE_JSON Atomically publish a UTF-8 metadata sidecar.
tmp=[tempname(fileparts(path)) '.json'];
fid=fopen(tmp,'w','n','UTF-8');
assert(fid>=0,'cells:JSON','Cannot open temporary metadata file.');
try
    fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
    fclose(fid); fid=-1;
    [ok,msg]=movefile(tmp,path,'f'); assert(ok,'cells:JSON','%s',msg);
catch e
    if fid>=0, fclose(fid); end
    if isfile(tmp), delete(tmp); end
    rethrow(e);
end
end
