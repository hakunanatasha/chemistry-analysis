
traj = traj_recompose[-1]

rdists = [(135, 142), (55, 58), (55, 57), (55, 56), (135, 58), (135, 57), (135, 56)]
ndists = len(rdists)
frames = np.zeros(shape = (Nframes, ndists))*np.nan

dcds = []
for fr in range(0, len(traj), 4):
    tset = traj[(fr):(fr+4)]
    formed_start, formed_end = [int(i) for i in tset[0].split(':')]
    calc_start, calc_end = [int(i) for i in tset[1].split(':')]
    irev = isrev[tset[2]]
    trajname = tset[3]

    if "prod" in trajname:
        trajname = "dcd/" + trajname + "_trim.dcd"

    if formed_start > formed_end:
        formed_start, formed_end = formed_end, formed_start

    if calc_start > calc_end:
        calc_start, calc_end = calc_end, calc_start

    if os.path.exists(trajname):
        
        dcd = get_trace(trajname, calc_start, calc_end, irev)

        r = np.array(list(map( lambda x: get_dist(dcd[:, x[0], :],dcd[:, x[1], :]), rdists)))

        if irev:
            cstart = r.shape[1] - (calc_end+1)
            cend   = cstart + (calc_end+1-calc_start)
            frames[formed_start-1:formed_end, :] = r[:, cstart:cend].T
        else:
            frames[formed_start-1:formed_end, :] = r[:, calc_start:(calc_end+1)].T


        #dcds.append(get_dist(dcd[:, 135, :],dcd[:, 142, :]))

        #for idx, dist in enumerate(rdists):
        #    frames[formed_start:formed_end, idx] = get_dist(dcd[:, dist[0], :], 
        #                                                    dcd[:, dist[1], :])
    else:
        print(trajname)



