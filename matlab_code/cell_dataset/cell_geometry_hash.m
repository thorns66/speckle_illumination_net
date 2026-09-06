function hash = cell_geometry_hash(folder)
%CELL_GEOMETRY_HASH Fingerprint the two files that define frozen truth.
if nargin<1 || isempty(folder), folder=fileparts(mfilename('fullpath')); end
md=java.security.MessageDigest.getInstance('SHA-256');
for file={'cell_dataset_config.m','cell_make_truth.m'}
    md.update(uint8(unicode2native(fileread(fullfile(folder,file{1})),'UTF-8')));
end
hash=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end
