function split = cell_dataset_split(sampleId)
%CELL_DATASET_SPLIT Authoritative whole-object split, independent of GT hash.
% Dataset membership is metadata and may change without changing frozen
% geometry, illumination, forward data, or reconstruction artifacts.
id=upper(char(sampleId));
trainIds={'P01','P02','P03','P04','P05','P06','P08','P10'};
validationIds={'P09','V01','V02'};
testIds={'P07','T01','T02'};
if ismember(id,trainIds)
    split='train';
elseif ismember(id,validationIds)
    split='validation';
elseif ismember(id,testIds)
    split='test';
else
    error('cells:Sample','Unknown complete-dataset sample ID %s.',id);
end
end
