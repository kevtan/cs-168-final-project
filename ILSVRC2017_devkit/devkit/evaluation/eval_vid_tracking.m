% this code is inspired by eval_vid_detection devkit
function [ap,recall,precision] = eval_vid_tracking(predict_file,gtruth_dir,meta_file,...
    eval_file,blacklist_file,optional_cache_file)
% Evaluate tracking in video
% - predict_file: each line is a single predicted object in the
%   format
%    <frame_id> <ILSVRC2015_VID_ID> <track_id> <confidence> <xmin> <ymin> <xmax> <ymax>
% - gtruth_dir: a path to the directory of ground truth information,
%   e.g. ../../Annotations/VID/val/
% - meta_file: information about the synsets
% - eval_file: list of images to evaluate on
% - blacklist_file: list of image/category pairs which aren't
%    considered in evaluation
% - optional_cache_file: to save the ground truth data and avoid
%    loading from scratch again

if nargin < 3
    meta_file = '../data/meta_vid.mat';
end
if nargin < 4
    eval_file = '../../ImageSet/VID/val.txt';
end
if nargin < 5
    blacklist_file = '';
end
if nargin < 6
    optional_cache_file = '';
end

defaultIOUthr = 0.5;
pixelTolerance = 10;
defaultTrackThr = [0.25, 0.5, 0.75];

load(meta_file);
hash = make_hash(synsets);

bLoadXML = true;
if ~isempty(optional_cache_file) && exist(optional_cache_file,'file')
    fprintf('eval_vid_tracking :: loading cached ground truth\n');
    t = tic;
    load(optional_cache_file);
    fprintf('eval_vid_tracking :: loading cached ground truth took %0.1f seconds\n',toc(t));
    if exist('gt_track_img_ids','var')
        bLoadXML = false;
    end
end
if bLoadXML
    fprintf('eval_vid_tracking :: loading ground truth\n');
    t = tic;
    
    [img_basenames,gt_img_ids] = textread(eval_file,'%s %d');
    vid_basenames = cellfun(@(x)x(1:23), img_basenames, 'UniformOutput', false);
    gt_vid_names = unique(vid_basenames);
    
    num_vids = length(gt_vid_names);
    gt_vid_range = zeros(2,num_vids);
    gt_track_labels = cell(1,num_vids);
    gt_track_bboxes = cell(1,num_vids);
    gt_track_thr = cell(1,num_vids);
    gt_track_img_ids = cell(1,num_vids);
    gt_track_generated = cell(1,num_vids);
    num_track_per_class = [];
    tic
    for v=1:num_vids
        if toc > 60
            fprintf('              :: on %i of %i\n',v,num_vids);
            tic;
        end
        img_ids = find(strcmp(vid_basenames, gt_vid_names{v}));
        gt_vid_range(:,v) = [min(img_ids), max(img_ids)];
        num_imgs = length(img_ids);
        tracks = [];
        num_tracks = 0;
        recs = cell(1,num_imgs);
        count = 0;
        for i=img_ids'
            count = count + 1;
            rec = VOCreadxml(sprintf('%s/%s.xml',gtruth_dir, ...
                img_basenames{i}));
            recs{count} = rec;
            if ~isfield(rec.annotation,'object')
                continue;
            end
            for j=1:length(rec.annotation.object)
                obj = rec.annotation.object(j);
                trackid = str2double(obj.trackid);
                c = get_class2node(hash, obj.name);
                if isempty(find(tracks == trackid, 1))
                    num_tracks = num_tracks + 1;
                    tracks = [tracks(:); trackid];
                    if length(num_track_per_class) < c
                        num_track_per_class(c) = 1;
                    else
                        num_track_per_class(c) = num_track_per_class(c) + 1;
                    end
                end
            end
        end
        if num_tracks == 0
            continue;
        end
        gt_track_labels{v} = ones(1,num_tracks) * -1;
        gt_track_bboxes{v} = cell(1,num_tracks);
        gt_track_thr{v} = cell(1,num_tracks);
        gt_track_img_ids{v} = cell(1,num_tracks);
        gt_track_generated{v} = cell(1,num_tracks);
        count = 0;
        for i=img_ids'
            count = count + 1;
            rec = recs{count};
            if ~isfield(rec.annotation,'object')
                continue;
            end
            for j=1:length(rec.annotation.object)
                obj = rec.annotation.object(j);
                trackid = str2double(obj.trackid);
                c = get_class2node(hash, obj.name);
                k = find(tracks == trackid);
                gt_track_img_ids{v}{k}(end+1) = i;
                if gt_track_labels{v}(k) == -1
                    gt_track_labels{v}(k) = c;
                else
                    if gt_track_labels{v}(k) ~= c
                        error('Find inconsistent label in a track!');
                    end
                end
                b = obj.bndbox;
                bb = str2double({b.xmin b.ymin b.xmax b.ymax});
                gt_track_bboxes{v}{k}(:,end+1) = bb;
                generated = str2double(obj.generated);
                gt_track_generated{v}{k}(end+1) = generated;
                gt_w = bb(4)-bb(2)+1;
                gt_h = bb(3)-bb(1)+1;
                thr = (gt_w*gt_h)/((gt_w+pixelTolerance)*(gt_h+pixelTolerance));
                gt_track_thr{v}{k}(end+1) = min(defaultIOUthr,thr);
            end
        end
    end
    fprintf('eval_vid_tracking :: loading ground truth took %0.1f seconds\n',toc(t));
    
    if ~isempty(optional_cache_file)
        fprintf('eval_vid_tracking :: saving cache in %s\n',optional_cache_file);
        save(optional_cache_file,'gt_img_ids','gt_vid_names','gt_vid_range',...
            'gt_track_labels','gt_track_bboxes','gt_track_thr','gt_track_img_ids',...
            'gt_track_generated','num_track_per_class');
    end
end

blacklist_img_id = [];
blacklist_label = [];
if ~isempty(blacklist_file) && exist(blacklist_file,'file')
    [blacklist_img_id,wnid] = textread(blacklist_file,'%d %s');
    blacklist_label = zeros(length(wnid),1);
    for i=1:length(wnid)
        blacklist_label(i) = get_class2node(hash,wnid{i});
    end
    fprintf('eval_vid_tracking :: blacklisted %i image/object pairs\n',length(blacklist_label));
else
    fprintf('eval_vid_tracking :: no blacklist\n');
end

fprintf('eval_vid_tracking :: loading predictions\n');
t = tic;
[img_ids,obj_labels,obj_track_ids,obj_confs,xmin,ymin,xmax,ymax] = ...
    textread(predict_file,'%d %d %d %f %f %f %f %f');
obj_bboxes = [xmin ymin xmax ymax]';
fprintf('eval_vid_tracking :: loading predictions took %0.1f seconds\n',toc(t));

fprintf('eval_vid_tracking :: sorting predictions\n');
t = tic;
[img_ids,ind] = sort(img_ids);
obj_labels = obj_labels(ind);
obj_track_ids = obj_track_ids(ind);
if any(obj_track_ids == -1)
    error('Find -1 in track id.');
end
obj_confs = obj_confs(ind);
obj_bboxes = obj_bboxes(:,ind);

num_vids = length(gt_vid_names);

track_img_ids = cell(1,num_vids);
track_labels = cell(1,num_vids);
track_confs = cell(1,num_vids);
track_bboxes = cell(1,num_vids);
tic
for v=1:num_vids
    if toc > 60
        fprintf('               :: on %d of %d\n',v,num_vids);
        tic
    end
    % retrieve results for current video.
    start_id = gt_vid_range(1,v);
    end_id = gt_vid_range(2,v);
    ind = img_ids >= start_id & img_ids <= end_id;
    vid_img_ids = img_ids(ind);
    vid_obj_labels = obj_labels(ind);
    vid_track_ids = obj_track_ids(ind);
    vid_obj_confs = obj_confs(ind);
    vid_obj_bboxes = obj_bboxes(:,ind);
    % get result for each tracklet in a video.
    track_ids = unique(vid_track_ids);
    num_tracks = length(track_ids);
    track_img_ids{v} = cell(1,num_tracks);
    track_labels{v} = ones(1,num_tracks) * -1;
    track_confs{v} = zeros(1,num_tracks);
    track_bboxes{v} = cell(1,num_tracks);
    count = 0;
    for k=track_ids'
        ind = vid_track_ids == k;
        count = count + 1;
        track_img_ids{v}{count} = vid_img_ids(ind);
        track_label = unique(vid_obj_labels(ind));
        if length(track_label) > 1
            error('Found multiple labels in a tracklet.');
        end
        track_labels{v}(count) = track_label;
        % use the mean score as a score for a tracklet.
        track_confs{v}(count) = mean(vid_obj_confs(ind));
        track_bboxes{v}{count} = vid_obj_bboxes(:,ind);
    end
end

for v=1:num_vids
    [track_confs{v}, ind] = sort(track_confs{v},'descend');
    track_img_ids{v} = track_img_ids{v}(ind);
    track_labels{v} = track_labels{v}(ind);
    track_bboxes{v} = track_bboxes{v}(:,ind);
end
tp_cell = cell(1,num_vids);
fp_cell = cell(1,num_vids);

fprintf('eval_vid_tracking :: sorting predictions took %0.1f seconds\n', ...
    toc(t));

fprintf('eval_vid_tracking :: accumulating\n');

num_classes = length(num_track_per_class);
num_track_thr = length(defaultTrackThr);

t = tic;
tic;
% iterate over videos
for v=1:num_vids
    if toc > 60
        fprintf('               :: on %d of %d\n',v,num_vids);
        tic;
    end
    
    num_tracks = length(track_labels{v});
    num_gt_tracks = length(gt_track_labels{v});
    
    tp = cell(1,num_track_thr);
    fp = cell(1,num_track_thr);
    gt_detected = cell(1,num_track_thr);
    for o=1:num_track_thr
        tp{o} = zeros(1,num_tracks);
        fp{o} = zeros(1,num_tracks);
        gt_detected{o} = zeros(1,num_gt_tracks);
    end
    
    for m=1:num_tracks
        img_ids = track_img_ids{v}{m};
        label = track_labels{v}(m);
        bboxes = track_bboxes{v}{m};
        num_obj = length(img_ids);
        
        ovmax = ones(1,num_track_thr) * -inf;
        kmax = ones(1,num_track_thr) * -1;
        for n=1:num_gt_tracks
            gt_label = gt_track_labels{v}(n);
            if label ~= gt_label
                continue;
            end
            gt_img_ids = gt_track_img_ids{v}{n};
            gt_bboxes = gt_track_bboxes{v}{n};
            gt_thr = gt_track_thr{v}{n};
            
            num_matched = 0;
            num_total = length(union(img_ids, gt_img_ids));
            for j=1:num_obj
                id = img_ids(j);
                k = find(gt_img_ids == id);
                if isempty(k)
                    continue;
                end
                if length(k) > 1
                    error('Find multiple ground truth.');
                end
                
                bSameImg = blacklist_img_id == id;
                blacklisted_obj = blacklist_label(bSameImg);
                
                if any(label == blacklisted_obj)
                    continue; % just ignore this detection
                end
                bb = bboxes(:,j);
                bbgt = gt_bboxes(:,k);
                bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
                iw=bi(3)-bi(1)+1;
                ih=bi(4)-bi(2)+1;
                if iw>0 && ih>0
                    % compute overlap as area of intersection / area of union
                    ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
                        (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
                        iw*ih;
                    % makes sure that this object is detected according
                    % to its individual threshold
                    ov=iw*ih/ua;
                    if ov >= gt_thr(k)
                        num_matched = num_matched + 1;
                    end
                end
            end
            ov = num_matched / num_total;
            for o=1:num_track_thr
                if gt_detected{o}(n)
                    continue;
                end
                if ov >= defaultTrackThr(o) && ov > ovmax(o)
                    ovmax(o) = ov;
                    kmax(o) = n;
                end
            end
        end
        for o=1:num_track_thr
            if kmax(o) > 0
                tp{o}(m) = 1;
                gt_detected{o}(kmax(o)) = 1;
            else
                fp{o}(m) = 1;
            end
        end
    end
    % put back into global vector
    tp_cell{v} = tp;
    fp_cell{v} = fp;
end
fprintf('eval_vid_tracking :: accumulating took %0.1f seconds\n', ...
    toc(t));

fprintf('eval_vid_tracking :: computing ap\n');
t = tic;
recall = cell(1,num_track_thr);
precision = cell(1,num_track_thr);
ap = cell(1,num_track_thr);
confs = [track_confs{:}];
[~, ind] = sort(confs,'descend');
for o=1:num_track_thr
    tp_all = [];
    fp_all = [];
    for v=1:num_vids
        tp_all = [tp_all(:); tp_cell{v}{o}'];
        fp_all = [fp_all(:); fp_cell{v}{o}'];
    end
    
    tp_all = tp_all(ind)';
    fp_all = fp_all(ind)';
    obj_labels = [track_labels{:}];
    obj_labels = obj_labels(ind);
    
    for c=1:num_classes
        % compute precision/recall
        tp = cumsum(tp_all(obj_labels==c));
        fp = cumsum(fp_all(obj_labels==c));
        recall{o}{c}=(tp/num_track_per_class(c))';
        precision{o}{c}=(tp./(fp+tp))';
        ap{o}(c) =VOCap(recall{o}{c},precision{o}{c});
    end
end
fprintf('eval_vid_tracking :: computing ap took %0.1f seconds\n', ...
    toc(t));
