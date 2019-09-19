import numpy as np
import scipy
import datetime
import multiprocessing as mp
import time

def get_wb(fnpz='../data/wbdict_5m.npz') :
    wb=np.load(fnpz)
    wb.allow_pickle=True
    return wb.items()[0][1].item()['wbar'][2:,:,:]

def get_lr_dt(wb=None, fnpz=None) :
    if wb is None :
        wb=get_wb(fnpz)
    lr=wb[:,:,1]
    dt=[]
    for t0 in wb[0,:,0] :
        dt.append(datetime.datetime.fromtimestamp(t0))
    dt=np.array(dt)
    return lr, dt

def get_lrd(lr_week) :
    n,m=lr_week.shape
    db=m/5
    return lr_week.reshape((n*m/db,db))

def py_sym_penta(d0, d1, d2) :
    """
    So the procedure would be
    1/a(1,1),       0, ..., 0 
    -a(2,1)/a(1,1), 1, ..., 0
    -a(3,1)/a(1,1), 0, ..., 0
    ...
    -a(n,1)/a(1,1), 0, ...  1
    """
    pass

dfn = ['norm','beta','gamma','dgamma','dweibull','cauchy','invgamma','invweibull','powerlaw','powerlognorm']
#dfn = ['norm']

def chisquare_test(x0,dn,param,bins=6) :
    """
    make each bin at least about 20 observations.  
    and at least 20 bins.  The more observations the better.
    That's important for CLT to work
    """
    n=len(x0)
    ixr=np.nonzero(np.abs(x0)<np.std(x0)*12)[0]
    x=x0[ixr].copy()
    x.sort()
    c=dn.cdf(x, *param[:-2], loc=param[-2], scale=param[-1])
    cs=np.linspace(c[0],c[-1], np.floor((c[-1]-c[0])*bins))
    ix=np.searchsorted(c, cs[1:]-1e-12)
    cnt=ix-np.r_[0,ix[:-1]]
    E=(c[ix]-c[np.r_[0,ix[:-1]]])*n
    v,p=scipy.stats.chisquare(cnt,E,ddof=len(param)-1)
    return v,p

def chisquare_test_unstable(x, dn, param) :
    n=len(x)
    cnt,bv=np.histogram(x,bins=min(1000, max(n/10, 5)))
    c1=dn.cdf(np.r_[bv[1:-1], 1e+14], *param[:-2], loc=param[-2], scale=param[-1])
    c2=np.r_[0,c1[:-1]]

    #c1=dn.cdf(bv[1:], *param[:-2], loc=param[-2], scale=param[-1])
    #c2=dn.cdf(bv[:-1], *param[:-2], loc=param[-2], scale=param[-1])

    xmid=(bv[1:]+bv[:-1])/2.0
    pc=cnt
    E=((c1-c2)*n).astype(int)

    # remove 0 in E
    zix=np.nonzero(E)[0]+1
    zix0=np.r_[0,zix[:-1]]
    cspc=np.r_[0,np.cumsum(pc)]
    pc=cspc[zix]-cspc[zix0]
    csE =np.r_[0,np.cumsum(E)]
    E=csE[zix]-csE[zix0]

    # remove the tail stuffs
    pc=pc[1:-1].astype(float)
    E=E[1:-1].astype(float)
    v,p=scipy.stats.chisquare(pc,E)
    return v,p

def distfit(x) :
    """
    run with chisquare
    """
    pks=[]
    pchi=[]
    dfs=['dgamma','dweibull','cauchy','norm']
    #dfs=dfn

    for df in dfs :
        try :
            d=getattr(scipy.stats,df)
            param=d.fit(x)
            v,p=scipy.stats.kstest(x,df,args=param)
            pks.append(p)
            v,p=chisquare_test(x,d,param)
            pchi.append(p)
        except KeyboardInterrupt as e:
            return
        except :
            print 'problem fitting ', df

    ps=np.array(pks) + np.array(pchi)
    ix=np.argsort(ps)[-1]
    print dfs[ix], pks[ix], pchi[ix]

    return pks, pchi
    #for dn, pk, pc in zip(dfn, pks, pchi) :
    #    print dn, pk, pc

def bootstrap_qr0(lr,cnt=None,m0=None,ixa=None,need_var=False) :
    """
    select subset of columns in lr for qr, in case
    n<m.  Note the bootstrap is not strict as
    the order is enforced and no duplicate is allowed
    # consider using a thread pool
    """
    n,m=lr.shape
    if n>m :
        q,r=np.linalg.qr(lr)
        return q,r,None,None
    if m0 is None:
        m0=n-1
    if cnt is None :
        cnt = int(200.0 * (float(m)/float(m0)))
    q=np.zeros((n,m))
    q2=np.zeros((n,m))
    r=np.zeros((m,m))
    r2=np.zeros((m,m))
    for c in np.arange(cnt) :
        if ixa is not None :
            ix=ixa[c]
        else :
            ix=np.random.choice(m,m0,replace=False)
        ix.sort()
        q0,r0=np.linalg.qr(lr[:,ix])
        # need to fix the sign in case the diagonal is negative
        for i,x in enumerate(np.diag(r0)):
            if x<0 :
                r0[i,:]*=-1
        q[:,ix]+=q0
        q2[:,ix]+=q0**2
        r[ np.ix_(ix,ix)]+=r0
        r2[np.ix_(ix,ix)]+=r0**2

    q/=cnt
    r/=cnt
    if not need_var: 
        return q,r
    q2/=cnt
    r2/=cnt
    return q, r, np.sqrt(q2-q**2), np.sqrt(r2-r**2)

def bootstrap_qr(lr,cnt=None,m0=None,njobs=8,ixa0=None,need_var=False) :
    """
    select subset of columns in lr for qr, in case
    n<m.  Note the bootstrap is not strict as
    the order is enforced and no duplicate is allowed
    # consider using a thread pool
    """
    n,m=lr.shape
    if n>m :
        q,r=np.linalg.qr(lr)
        if not need_var :
            return q,r
    if m0 is None:
        m0=min(n-1,m*0.8)  # bootstrap size
    if cnt is None :
        cnt = int(64.0 * (float(m)/float(m0)))
    q=np.zeros((n,m))
    q2=np.zeros((n,m))
    r=np.zeros((m,m))
    r2=np.zeros((m,m))
    while True :
        try :
            pool=mp.Pool(processes=njobs)
            break
        except :
            print 'problem with resource, sleep a while'
            time.sleep(5)

    results=[]
    ixa=[]
    for c in np.arange(cnt) :
        if ixa0 is not None:
            ix=ixa0[c]  
        else :
            ix=np.random.choice(m,m0,replace=False)
        ix.sort()
        results.append((c, ix.copy(), pool.apply_async(np.linalg.qr,args=(lr[:,ix].copy(),))))

        if (c+1)%njobs == 0 or c+1==cnt:
            for res0 in results :
                c0,ix,res=res0
                q0,r0=res.get()
                #q0,r0=np.linalg.qr(lr[:,ix])
                # need to fix the sign in case the diagonal is negative
                for i,x in enumerate(np.diag(r0)):
                    if x<0 :
                        r0[i,:]*=-1
                q[:,ix]+=q0
                q2[:,ix]+=q0**2
                r[ np.ix_(ix,ix)]+=r0
                r2[np.ix_(ix,ix)]+=(r0**2)
                ixa.append(ix.copy())

            results=[]
            print 'iteration ', c
    pool.close()

    q/=cnt
    r/=cnt
    if not need_var:
        return q,r
    q2/=cnt
    r2/=cnt
    #return q,r,q2,r2
    return q, r, np.sqrt(q2-q**2), np.sqrt(r2-r**2)

def mergelr(lr, frac, dt=None, ix0=None) :
    """
    lr0 = mergelr(lr, frac)
    lr shape [n,m], lr0 shape [n,m0]
    m0=frac*m, frac in [0,1]
    reduce the number of bars (sample
    points)
    """
    n,m=lr.shape
    mm = m*frac
    if ix0 is not None:
        ix=np.array(ix0).copy()
    else :
        ix = np.arange(m)
    lrc=np.vstack((np.zeros(n),np.cumsum(lr,axis=1).T)).T
    qr_score=[]
    qr_remove=[]
    while len(ix) > mm :
        #lr0=lrc[:,ix+1]-lrc[:,ix]
        lr0=lrc[:,ix+1]-lrc[:,np.r_[0,ix[:-1]+1]]
        vol=np.std(lr0,axis=0)
        n0=len(vol)

        print 'currently ', n0, ' removing one...'
        lr00=lr0/vol
        q0,r0=bootstrap_qr(lr00)

        #r0=np.abs(r0)
        #wt=(r0[0,0]-r0.diagonal())*vol
        r0=np.abs(r0)
        wt=vol

        print 'tot vol (sum,mean) ', np.sum(vol), np.mean(vol), 'avg of noise ', np.mean(r0.diagonal()**2), ' avg of signal ', np.sum(wt), np.mean(wt)

        r0*=wt
        r0=r0+r0.T
        snr=np.sum(r0,axis=1)-r0.diagonal()
        ix0_=np.argsort(snr)  # getting the least useful bar
        print 'lowest signal contributors ', ix[ix0_[:10]], snr[ix0_[:10]]
        qr_score.append(snr[ix0_[0]])
        if dt is not None:
            print dt[ix[ix0_[:10]]]

        ix1_list=[]
        for ix0 in ix0_[:10]:
        #ix0=ix0_[0]
            if ix0 == 0 :
                ix1_list.append(ix0)
            elif ix0 == n0-1 :
                ix1_list.append(ix0-1)
            else :
                ix1_list.append(ix0-1)
                ix1_list.append(ix0)

        ix1_list=np.unique(ix1_list)
        rscore=[]
        for ix1 in ix1_list:
            ix_ = np.delete(ix, ix1)
            #lr00 = lrc[:,ix_+1]-lrc[:,ix_]
            lr00 = lrc[:,ix_+1]-lrc[:,np.r_[0,ix_[:-1]+1]]
            q_,r_=bootstrap_qr(lr00,cnt=30)

            # note the total signal (weighted by vol)
            # is the goal

            r_=r_**2
            rd=np.abs(r_.diagonal())
            #rs0=np.sum(rd)*np.sqrt(np.dot(rd,rd))
            rs0=np.sum(np.sqrt(np.sum(r_,axis=0)-r_.diagonal()))
            rs1=np.sqrt(np.mean(r_.diagonal())-np.mean(rd)**2)

            print 'rs numbers: total signal, variance of signal, sharp', rs0, rs1, rs0/rs1
            rs0/=rs1
            #rs0=np.sum(rd)  # ignoring the variance in rs0 for now
            dtstr=''
            if dt is not None:
                dtstr=dt[ix[ix1]]
            print '** Score ', ix1, dtstr, rs0
            rscore.append(rs0)

        # find the ix with best score
        rsix=np.argsort(rscore)[::-1]
        print 'gains for removing the bar:'
        for rsix0 in rsix :
            ix1=ix1_list[rsix0]
            dtstr='' if dt is None else dt[ix[ix1]]
            print '   ', dtstr, ix[ix1], rscore[rsix0]

        # go with the biggest sharp
        ix1=ix1_list[rsix[0]]

        # remove ix1
        print 'So, removing ', ix[ix1], '' if dt is None else dt[ix[ix1]]
        qr_remove.append(ix[ix1])
        ix=np.delete(ix,ix1)
    return ix, qr_score, qr_remove

def wtD1(n) :
    pass

def wtD2(n) :
    pass


