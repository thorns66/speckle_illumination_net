function pilot_atomic_save(path, payload)
%PILOT_ATOMIC_SAVE Publish one MAT artifact only after a complete save.
assert(isstruct(payload) && isscalar(payload), 'pilot:SavePayload', ...
    'payload must be a scalar struct.');
parent = fileparts(path);
assert(isfolder(parent), 'pilot:SaveParent', 'Missing output directory: %s', parent);
tmp = [tempname(parent) '.mat'];
cleanup = onCleanup(@() cleanup_tmp(tmp)); %#ok<NASGU>
save(tmp, '-struct', 'payload', '-v7.3');
[ok, message] = movefile(tmp, path, 'f');
assert(ok, 'pilot:SavePublish', 'Cannot publish %s: %s', path, message);
end

function cleanup_tmp(path)
if isfile(path), delete(path); end
end
