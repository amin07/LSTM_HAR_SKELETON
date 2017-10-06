function sampled_seq = preprocess_sequence(filename, T, considered_joints)
% Reads an .skeleton file from "NTU RGB+D 3D Action Recognition Dataset".
% 
% Argrument:
%   filename: full path to the skeleton file
%   T: number of samples to be taken from the sequence
%   joints: joint list need to be considered
% Returns:
% A matrix of dimension joints X T X 3

if nargin<3
    T = 10;
    considered_joints = [6,10,22,24];
elseif nargin < 4
    considered_joints = [6,10,22,24];
end
    
shoulder_spines = [5,9,1,2];

bodyinfo = read_skeleton_file(filename);
frame_count = size(bodyinfo, 2);

frame_count = frame_count - mod(frame_count, T);
% fprintf('frame count after mod %d\n', frame_count);
window_size = frame_count/T;
num_joints = size(considered_joints, 2);
sampled_seq = ones(num_joints, T, 3);


% getting the (left, right shoulder) and (spine_base, spine) for adjusting
% xy plane purpose taking value from the first frame (most stable)

try
    left_shoulder = bodyinfo(1).bodies(1).joints(shoulder_spines(1));
    right_shoulder = bodyinfo(1).bodies(1).joints(shoulder_spines(2));
    spine_base = bodyinfo(1).bodies(1).joints(shoulder_spines(3));
    spine_mid = bodyinfo(1).bodies(1).joints(shoulder_spines(4));


    % this are for rotation purpose, but rotation is not done yet
    left_shoulder = [left_shoulder.x, left_shoulder.y, left_shoulder.z];
    right_shoulder = [right_shoulder.x, right_shoulder.y, right_shoulder.z];
    spine_base = [spine_base.x, spine_base.y, spine_base.z];
    spine_mid = [spine_mid.x, spine_mid.y, spine_mid.z];




    j_count = 1;
    for i=1:window_size:frame_count
        rand_frame = randi([i, i+window_size-1]);
        %rand_frame = i;         %only for validation purpose
        joints = bodyinfo(rand_frame).bodies(1).joints;
        for j=1:numel(considered_joints)
    %         fprintf('joint id %d\n',considered_joints(j));
            jt = joints(considered_joints(j));
            sampled_seq(j,j_count,:) = [jt.x, jt.y, jt.z];
        end
        j_count = j_count + 1;


    end

    reshaped_sample = reshape(sampled_seq, [num_joints*T, 3]);
    origin_changed = bsxfun(@minus, reshaped_sample, spine_mid);
    sampled_seq = reshape(origin_changed, [num_joints, T, 3]);
catch
    disp('Missing joints exceptions');
    sampled_seq = [];
end

end