!*************************************************************************************************!
!                                                                                                 !
Module Greenf                                                                                    !
!                                                                                                 !
!*************************************************************************************************!
Use Define
Use rhflib
Use DiracR
Use density 
use omp_lib
implicit none

double precision :: er_step, ecut, er_preci, hgam_max, Cst
data er_step/5.d-3/, ecut/32.d0/, er_preci/1.d-7/



!--- Green's Function Non-local density by gaow 2024.11
    type GreenFun
    double precision, dimension(MSD, MSD) :: gg, gf, ff
    double precision, dimension(MSD) :: Hgg, Hgf, Hff
    end type GreenFun
    type (GreenFun), dimension(NTX, 2) :: ResGF

    type Green
    complex*16 , dimension(MSD, MSD), private :: Fgg, Fgf, Fff
    complex*16 , dimension(MSD, MSD), private :: Res_gg, Res_gf, Res_ff
    complex*16 , dimension(MSD), private :: Res_Hgg, Res_Hgf, Res_Hff, Hgf
    integer :: i0, it, kappa, node, ib
    end type Green
    type (Green):: GrF

    type Density_of_States
    complex*16 , dimension(MSD) :: Hgg, Hff
    end type Density_of_States
    type (Density_of_States):: DOS
    double precision, allocatable :: Ers(:), densta(:)
!$OMP THREADPRIVATE(DOS)   
Contains        
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
Subroutine densitGF                                                                         !
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!--- To calcualte the densities
double precision :: h, rel, xvv, r2
double precision :: ftemp, gtemp, zeta, gftemp
integer :: it, i, npt, i0, ib, n, nd, ka
double precision :: epsilon, g, e1, e2, C, prec, gv, eh, fun(MSD)
complex*16 Ec, ui
!use, intrinsic :: ieee_arithmetic
 
ui=(0.d0, 1.d0)
npt = well%npt
h   = well%h
do it = 1, 2
    if (pair%is(it).ne.1) cycle
    do ib = 1, chunk(it)%nb
        ka  = chunk(it)%kbl(ib)
        !if (abs(ka).le.2) cycle
        do n = 1, chunk(it)%id(ib)
            i0  = chunk(it)%ia(ib) + n
            if (.not.lev(it)%lpG(i0))cycle
            !
            if(lev(it)%hgam(i0).gt.hgmin) then
                !--- Calculate XG, XF, YG, YF
                if(IE.eq.2) call GFDetXY(i0,it,.false.)  
            write(*,'(2a10,f12.6,a10,2f12.6)')lev(it)%tb(i0),'hgam=',lev(it)%hgam(i0),'ee=',lev(it)%ee(i0),hgmin
                eh=8.*lev(it)%hgam(i0)/200
                prec=lev(it)%ee(i0)-4.*lev(it)%hgam(i0)
                ResGF(i0,it)%Hgg=zero; ResGF(i0,it)%Hff=zero; ResGF(i0,it)%Hgf=zero
                ResGF(i0,it)%gg=zero; ResGF(i0,it)%ff=zero; ResGF(i0,it)%gf=zero
                do i=0, 200
                    epsilon=i*eh+prec
                    g=g_alh(epsilon, lev(it)%hgam(i0), lev(it)%ee(i0))
                    e1=epsilon-fermi%ala(it) 
                    e2=one/dsqrt(e1**2+lev(it)%del(i0)**2)
                    gv=half*g*(one-e1*e2)*eh!/gc(it,i0)/lev(it)%vv(i0)
           
                    Ec = epsilon-ui*1.d-6; C=1.d0
                    call DiracGF(it,ka,Ec,non_local)
                    fun = -aimag(DOS%HGG) - aimag(DOS%HFF)
                    call simps(fun,npt,well%h, C)
                    !if (ieee_is_nan(C))cycle
                    ResGF(i0,it)%Hgg=ResGF(i0,it)%Hgg-gv*aimag(DOS%HGG)/C
                    ResGF(i0,it)%Hff=ResGF(i0,it)%Hff-gv*aimag(DOS%HFF)/C     
                    ResGF(i0,it)%Hgf=ResGF(i0,it)%Hgf-gv*aimag(GrF%Hgf)/C
                    if(non_local)then
                        ResGF(i0,it)%gg=ResGF(i0,it)%gg-gv*aimag(GrF%Fgg)/C
                        ResGF(i0,it)%ff=ResGF(i0,it)%ff-gv*aimag(GrF%Fff)/C
                        ResGF(i0,it)%gf=ResGF(i0,it)%gf-gv*aimag(GrF%Fgf)/C     
                    end if
                end do
                fun =ResGF(i0,it)%Hgg+ResGF(i0,it)%Hff
                    call simps(fun,npt,well%h, C)
                    write(*,'(2a10,4f12.6)')lev(it)%tb(i0),'C=',C, lev(it)%vv(i0), C/lev(it)%vv(i0)
                    ResGF(i0,it)%Hgg=ResGF(i0,it)%Hgg/C
                    ResGF(i0,it)%Hff=ResGF(i0,it)%Hff/C     
                    ResGF(i0,it)%Hgf=ResGF(i0,it)%Hgf/C
                    if(non_local)then
                        ResGF(i0,it)%gg=ResGF(i0,it)%gg/C
                        ResGF(i0,it)%ff=ResGF(i0,it)%ff/C
                        ResGF(i0,it)%gf=ResGF(i0,it)%gf/C     
                    end if
            end if
        end do
    end do
end do
do it = 1, 2
do nd = 2, npt
    dens(it)%rs(nd) = zero
    dens(it)%rv(nd) = zero
    dens(it)%rt(nd) = zero
    do ib = 1, chunk(it)%nb
        ka  = chunk(it)%kbl(ib)
        do n = 1, chunk(it)%id(ib)
            i0  = chunk(it)%ia(ib) + n
            rel = lev(it)%vv(i0)*lev(it)%mu(i0)
            dens(it)%rs(nd) = dens(it)%rs(nd) + (ResGF(i0,it)%Hgg(nd) - ResGF(i0,it)%Hff(nd))*rel
            dens(it)%rv(nd) = dens(it)%rv(nd) + (ResGF(i0,it)%Hgg(nd) + ResGF(i0,it)%Hff(nd))*rel
            dens(it)%rt(nd) = dens(it)%rt(nd) + two*ResGF(i0,it)%Hgf(nd)*rel
        end do
    end do
    dens(it)%rs(nd) = dens(it)%rs(nd)/(4.*pi*well%xr(nd)**2)
    dens(it)%rv(nd) = dens(it)%rv(nd)/(4.*pi*well%xr(nd)**2)
    dens(it)%rt(nd) = dens(it)%rt(nd)/(4.*pi*well%xr(nd)**2)
end do

dens(it)%rs(1)  = 3.*(dens(it)%rs(2) - dens(it)%rs(3)) + dens(it)%rs(4)  !r=0�ĵ㣬
dens(it)%rv(1)  = 3.*(dens(it)%rv(2) - dens(it)%rv(3)) + dens(it)%rv(4)
dens(it)%rt(1)  = 3.*(dens(it)%rt(2) - dens(it)%rt(3)) + dens(it)%rt(4)
end do

!--- To calculate rho and rho3
do nd = 1, npt
den%rs(nd)  = dens(1)%rs(nd) + dens(2)%rs(nd);      den%rs3(nd) = dens(1)%rs(nd) - dens(2)%rs(nd)
den%rv(nd)  = dens(1)%rv(nd) + dens(2)%rv(nd);      den%rv3(nd) = dens(1)%rv(nd) - dens(2)%rv(nd)
den%rt(nd)  = dens(1)%rt(nd) + dens(2)%rt(nd);      den%rt3(nd) = dens(1)%rt(nd) - dens(2)%rt(nd)
end do

!--- Calculate the non-local density for Fock terms
if(IE.eq.2) then
do it = 1, 2
    do ib = 1, chunk(it)%nb
        denf(ib,it)%gg  = zero;             denf(ib,it)%ff  = zero
        denf(ib,it)%gf  = zero;             denf(ib,it)%fg  = zero
        
        do n = 1, chunk(it)%id(ib)
            i0  = chunk(it)%ia(ib) + n
            xvv = lev(it)%vv(i0)*lev(it)%mu(i0)
        !$OMP PARALLEL DO private(i) SCHEDULE(STATIC)
            do nd = 1, npt
            do i = 1, npt
                denf(ib,it)%gg(nd,i)    = denf(ib,it)%gg(nd,i) + ResGF(i0,it)%gg(nd,i)*xvv
                denf(ib,it)%ff(nd,i)    = denf(ib,it)%ff(nd,i) + ResGF(i0,it)%ff(nd,i)*xvv

                denf(ib,it)%gf(nd,i)    = denf(ib,it)%gf(nd,i) + ResGF(i0,it)%gf(nd,i)*xvv
                denf(ib,it)%fg(nd,i)    = denf(ib,it)%fg(nd,i) + ResGF(i0,it)%gf(i,nd)*xvv
            end do                        
            end do
        !$OMP END PARALLEL DO
        end do
    end do
end do
end if
if(pset%inl.eq.0) then
!--- CALCULATE THE COUPLING CONSTANTS WITH RESPECT TO THE BARYONIC DENSITY
!$OMP PARALLEL DO private(zeta, gtemp, ftemp, gftemp) SCHEDULE(STATIC)
do n = 1, npt
zeta        =  den%rv(n)/pset%rvs

gtemp       =  (one + pset%bsig*(zeta + pset%dsig)**2)
ftemp       =  (one + pset%csig*(zeta + pset%dsig)**2)
gftemp      =  (pset%bsig - pset%csig)*(zeta + pset%dsig)

cct%gsig(n) =  pset%gsig*pset%asig* gtemp/ftemp
cct%dsig(n) =  pset%gsig*pset%asig*gftemp/ftemp**2*two

gtemp       =  (one + pset%bome*(zeta + pset%dome)**2)
ftemp       =  (one + pset%come*(zeta + pset%dome)**2)
gftemp      =  (pset%bome - pset%come)*(zeta + pset%dome)

cct%gome(n) =  pset%gome*pset%aome* gtemp/ftemp
cct%dome(n) =  pset%gome*pset%aome*gftemp/ftemp**2*two

gtemp       =  dexp(pset%arho*zeta)
cct%grho(n) =  pset%grho/gtemp
cct%drho(n) = -pset%arho*cct%grho(n)

gtemp       =  dexp(pset%artn*zeta)
cct%grtn(n) =  pset%grtn/gtemp
cct%drtn(n) = -pset%artn*cct%grtn(n)

gtemp       =  dexp(pset%apio*zeta)
cct%fpio(n) =  pset%fpio/gtemp
cct%dpio(n) = -pset%apio*cct%fpio(n)
end do
!$OMP END PARALLEL DO
else
!--- In the case of non-linear self-couplings
    cct%gsig    = pset%gsig;            cct%gome    = pset%gome;            cct%grho    = pset%grho
    cct%dsig    = zero;                 cct%dome    = zero;                 cct%drho    = zero
    
    cct%fpio    = pset%fpio;            cct%grtn    = pset%grtn
    cct%dpio    = zero;                 cct%drtn    = zero
end if
    return

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
End Subroutine densitGF                                                                             ! 
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
Subroutine GF_Detgff(lprx)                                                                        !
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!--- Solve the Dirac equation with local and non-local potentials in coordinate space
!--- Integro-differential Dirac equation is transformed into equivalent differential one
!--- Shoot method is applied in sovling the equations
!
    logical :: lprx, lpr_Res
    integer :: ib, i0, n0, ka, nmax, n, n_2, it, i, ip, Nest, Nest1
    double precision :: eig, epoint, esi, prec, er
    double precision :: Ebot = -80.d0 
    double precision, dimension(2) :: tc
    data tc/0.0d0, 1.0d0/
    data esi/0.001d0/
    n_2=5.d0/(er_step/10.d0)
    Nest=n_2+(ecut-5.d0)/er_step
    
    XF=zero ;   XG=zero;  YF=zero;     YG=zero
    XFh=zero;   XGh=zero;   YFh=zero;   YGh=zero
    Call intpol6(dself%cou,  dself%couh,  well%npt)
!--- Loop over neutron and proton
    do it = 1, 2
        Cst=tc(it)*dself%cou(well%npt)*hbc !Continuous spectrum threshold
        GrF%it  = it
    !--- Loop over the single particle levels
        do ib = 1, chunk(it)%nb
            n0          = chunk(it)%ia(ib)
            nmax        = chunk(it)%id(ib)
            ka  = chunk(it)%kbl(ib) 
            GrF%ib     = ib
            GrF%kappa  = ka
            
            do n = 1, nmax
                i0              = n0 + n
                eig             = lev(it)%ee(i0)
                GrF%node        = n
                GrF%i0          = i0
                !--- Calculate XG, XF, YG, YF
                if(IE.eq.2) call GFDetXY(i0,it,.false.)   
                lev(it)%lpG(i0) =.false.;   lev(it)%hgam(i0)=0.d0
                !--- Finding the eigenvalue
                    lpr_Res = .false.
                    if (eig .lt. Cst+0.5d0) call GF_bound_state(eig, er)
                    if (eig .gt. Cst .and. .not. lev(it)%lpG(i0)) then
                        lpr_Res = .true.
                        prec = eig - my_round(eig, 3)
                    end if
                if (lpr_Res) then
                    allocate(Ers(0:Nest))
                    ers(0) = er_preci  
                !$OMP PARALLEL DO PRIVATE(i)
                    do i=1,Nest
                        if (i.le.n_2)THEN
                            ers(i) = i*er_step/10+prec
                        else
                            Ers(i) = (i-n_2)*er_step + 5.0+prec
                        end if
                    end do
                !$OMP END PARALLEL DO
                    if (lprx .and. lprNP(it)) then
                        call GF_Reson_state(Nest)
                    elseif (pair%del(it) .ne. 0.d0) then
                        call GF_Reson_state(Nest)
                    elseif (fermi%ala(it) .ge. 0.d0) then
                        Nest1 = n_2+0.5*ecut/er_step
                        call GF_Reson_state(Nest1)
                    end if
                    deallocate(Ers)
                    exit
                end if
            end do !n
        end do !kap
    end do!it
    
    return
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
End Subroutine GF_Detgff                                                                          !
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
integer function find_closest(e,Nest)
    implicit none
    integer, intent(in) :: Nest
    double precision, intent(in) ::  e
    integer :: low, high, mid
    
    low = 1
    high = Nest
    do while (low < high)
        mid = (low + high) / 2
        if (ers(mid) >= e) then
            high = mid
        else
            low = mid + 1
        end if
    end do

    ! 确定最终位置
    if (low == 1) then
        find_closest = 1
    else if (e - ers(low-1) < ers(low) - e) then
        find_closest = low - 1
    else
        find_closest = low
    end if
end function
subroutine GF_ErDOS(n_begin,Nest,ResEl,ResEr,nmax, max_pos, lpr)
    integer, intent(in) :: n_begin, Nest
    integer, intent(out) :: nmax, max_pos
    double precision, intent(out) :: ResEl,ResEr
    logical, intent(out) :: lpr
    integer :: it, ka, n, i, nn, j, nmin,i0
    double precision :: ei, DeltaE, nt, free, emaxl, estpne
    double precision :: C, E_min, E_max, esip, erold, eiold, den_min

    
    it=GrF%it; i0=GrF%i0;  ka=GrF%kappa;   den_min=1.d-3; i=0; lpr=.false.
    densta=0.d0
    do n=n_begin,Nest
        
        nt = abs(CLD(it,ka,ers(n),1.d-6))
        free = - CLD0(it,ka,ers(n),1.d-6)
        densta(n) = nt - free
        if (i.eq.0.and.densta(n).gt.den_min)then
            nmin=n
            i=1
            E_min=ers(n)
        end if

        if (densta(n-1).gt.den_min)THEN
            if (densta(n).le.den_min .or. n.eq.Nest)THEN
                nmax=n
                E_max=ers(n)  
                exit
            end if
        endif  
    end do
    max_pos = maxloc(densta, dim = 1)-1
    if (max_pos.eq.Nest.or.densta(max_pos).le.den_min)return 
    if (densta(max_pos).gt.50.d0) goto 20
    do nn=nmin,nmax
        if (densta(nn).gt.densta(max_pos)*0.3)THEN
            if(i==1)then
                E_min=ers(nn-1)
                i=2
            end if
        elseif(i.eq.2)then
            E_max=ers(nn)
            exit
        end if
    end do
20  ResEr=E_max; ResEl=E_min; i=0 
    hgam_max=2.*(ResEr-ResEl)
    if (hgam_max.gt.Gam_max)hgam_max=Gam_max
    lpr=.true.    
    return
End subroutine GF_ErDOS 
subroutine GF_Reson_state(Nest)
    integer :: it, i0, ib, ka, n, nc, max_pos, Nest, nmin, nmax, j, nmin2,nmax2, nmax3
    logical :: lpr
    double precision :: ei, ResEl, ResEr, DeltaE, nt, free, emaxl, estpne
    double precision :: C, E_min, E_max, esip, erold, eiold, esip_old

    it=GrF%it; ib=GrF%ib; nc=GrF%node; i0=GrF%i0; ka=GrF%kappa;  nmin=1

    allocate(densta(0:Nest))
    nt = abs(CLD(it,ka,ers(0),1.d-6))
    free = - CLD0(it,ka,ers(0),1.d-6)
    densta(0) = nt - free
10  call GF_ErDOS(nmin,Nest,ResEl,ResEr,nmax, max_pos,lpr)
    if (.not.lpr) goto 12 
    write(*,111)lev(it)%tb(i0),'E_min,E_max=',ResEl,ResEr,'  Er,Γmax,DOS=',ers(max_pos),2*hgam_max,densta(max_pos)
        call Reson_state(ResEl,ResEr,erstep)
        if(IE.eq.2.and.lev(it)%lpG(i0)) then
            esip=1.d0;  j=0
            do while(esip.gt.0.0001d0 .and.j.lt.30)
                erold=lev(it)%ee(i0); eiold=lev(it)%hgam(i0);j=j+1
                call GFDetXY(i0,it,.false.)
                nmin2=nmin; nmax2=nmax
                if(eiold.lt.4)nmin2=find_closest(erold-4*eiold,Nest)
                nmax2=find_closest(erold+4*eiold,Nest)
                write(*,'(2a10,4f12.6)')lev(it)%tb(i0),'erold,eiold=',erold,eiold,Ers(nmin2),Ers(nmax2)
                call GF_ErDOS(nmin2,nmax2,ResEl,ResEr,nmax3, max_pos,lpr)
                write(*,111)lev(it)%tb(i0),'22E_min,E_max=',ResEl,ResEr,'Er,hgam_max,DOS=',ers(max_pos),hgam_max,densta(max_pos)
                call Reson_state(ResEl,ResEr,erstep)
                    esip_old=esip
                    esip=max(abs(lev(it)%ee(i0)-erold),abs(lev(it)%hgam(i0)-eiold))
                    !write(*,'(2a16,3f8.4)')lev(it)%tb(i0),'er,ei,esip=',lev(it)%ee(i0), lev(it)%hgam(i0), esip
                if (esip.gt.400 .or. esip.eq.esip_old) then
                    write(*,*)lev(it)%tb(i0), 'error in GF_Reson_state'
                    lev(it)%lpG(i0) =.false.;   lev(it)%hgam(i0)=0.d0
                    call GFDetXY(i0,it,.false.)
                    call Reson_state(ResEl,ResEr,erstep)
                    esip=5.d0
                    exit
                end if
            end do
            lev(it)%CC(i0)=esip
            if (esip.gt.0.0001d0) then
                write(*,*)lev(it)%tb(i0), erold, eiold, 'error in GF_Reson_state, esip=', esip
                lev(it)%lpG(i0) =.false.;   lev(it)%hgam(i0)=0.d0
                call GFDetXY(i0,it,.false.)
            end if
        end if
        
        nmin=nmax;  densta=0.d0
        if (lev(it)%lpG(i0))THEN
            write(*,'(2a16,2f8.4)')lev(it)%tb(i0),'er,ei=',lev(it)%ee(i0), lev(it)%hgam(i0)
            i0=i0+1; nc=nc+1;    E_max=ers(nmax)
            GrF%node=nc; GrF%i0=i0
           ! if (nc.gt.chunk(it)%id(ib)) goto 12
            if (IE.eq.2) call GFDetXY(i0,it,.false.)  
        end if
       if(Nest.gt.nmax) goto 10
12   deallocate(densta)
    return
111     format(a10,a16,2f8.4,a14,3f10.4,I4)
112     format(a10,3f12.6,I3,a16,2f8.4,a14,f8.4)
End subroutine GF_Reson_state  

subroutine Reson_state(ResEl,ResEr,estpne)
    integer :: it, i0, ka
    double precision, intent(in) :: ResEl,ResEr,estpne
    logical :: Erlpr, Eilpr
    double precision :: ermin,ermax,estp
    integer :: n, Nest, Nest2, max_pos, ni, mi
    double precision :: er, ei, C, dens, free
    double precision, allocatable :: denstas(:), hgam(:)
    
    it=GrF%it;  ka=GrF%kappa;   i0=GrF%i0
    lev(it)%lpG(i0) =.false.;   lev(it)%hgam(i0)=0.d0
    Nest =(ResEr-ResEl)/estpne+1
    Erlpr=.true.; Eilpr=.true.

    do ni=1, 4
        if (.not.Eilpr) goto 10
        allocate(denstas(0:Nest),hgam(0:Nest))
        denstas=0.d0
        mi = count_leading_zeros(estpne)
        !$OMP PARALLEL DO PRIVATE(n, Er)
            do n=0,Nest
                Er=ResEl + n*estpne
                call bisection(er, mi, ni, hgam(n), denstas(n))
            end do
        !$OMP END PARALLEL DO
            max_pos = maxloc(denstas, dim = 1)-1
            er=ResEl + max_pos*estpne
            ermin=er-5.d0*estpne
            if (ermin.lt.0.d0) ermin=1.d-10
            estp=estpne/10
        deallocate(denstas, hgam)
        free = abs(CLD0(it,ka,er,ei))
        !write(6,'(2a10,I2,2f12.6,2e12.5,f5.1)')lev(it)%tb(i0),'estpne,ni=',ni,er,ei,dens,free,dens/free
        Nest2 =10.d0*estpne/estp+1
        allocate(denstas(1:Nest2),hgam(1:Nest2))
            do while (estp.ge.er_preci)
            denstas=0.d0
            !$OMP PARALLEL DO PRIVATE(n, Er)
                do n=1,Nest2
                    Er=ermin + n*estp 
                    call bisection(er, mi, ni, hgam(n), denstas(n)) 
                end do
            !$OMP END PARALLEL DO
                max_pos = maxloc(denstas, dim = 1)
                er=ermin + max_pos*estp
                ermin=er-5.d0*estp
                if (ermin.lt.0.d0) ermin=1.d-10
                estp=estp/10; mi = mi + 1
            end do
            ei = hgam(max_pos)
            dens= denstas(max_pos)
        deallocate(denstas, hgam)
        free = abs(CLD0(it,ka,er,ei))
        if(dens/free.gt.10) write(6,'(2a10,I2,2f12.6,2Es12.5,f5.1)')lev(it)%tb(i0),'ni=',ni,er,ei,dens,free,dens/free
        if (dens/free.gt.10.and.dens.gt.2.d0) then
            call Check_density(er,ei,dens,C)
            return
        end if
        if (dens==0.d0) Eilpr=.false.
10      if (Erlpr) call Er_Reson_state(ResEl,ResEr,ni,Erlpr)
        if (lev(it)%lpg(i0)) return
    end do
    
End subroutine Reson_state
subroutine Er_Reson_state(ResEl,ResEr,ni,Erlpr)
    integer, intent(in) :: ni
    double precision, intent(in) :: ResEl,ResEr
    logical,intent(inout) :: Erlpr
    double precision :: eimin,eimax,estp,estpne
    integer :: it, i0, n, Nest, Nest2, ka, max_pos, mi
    double precision :: er, ei, C, dens, free
    double precision, allocatable :: denstas(:), Ern(:)
    it=GrF%it;  i0=GrF%i0
    estpne=erstep
    Nest =(hgam_max-1.d0)/estpne+1
        allocate(denstas(0:Nest),Ern(0:Nest))
        denstas=0.d0
        mi = count_leading_zeros(estpne)
        !$OMP PARALLEL DO PRIVATE(n, Er)
            do n=0,Nest
                Ei=n*estpne + er_preci +1.0
                call Er_bisection(ei, mi, ResEl, ResEr, ni, Ern(n), denstas(n))
            end do
        !$OMP END PARALLEL DO
            max_pos = maxloc(denstas, dim = 1)-1
            ei=max_pos*estpne + er_preci + 1.0
            eimin=ei-2.d0*estpne
            estp=estpne/10
            er = Ern(max_pos)
            densta= denstas(max_pos)   
        deallocate(denstas,Ern)
       if (er.eq.0.d0) return
        Nest2 =4.d0*estpne/estp+1
        allocate(denstas(1:Nest2),Ern(1:Nest2))
        do while (estp.ge.er_preci)
        denstas=0.d0; mi = mi + 1
        if(eimin.lt.0.d0) eimin=er_preci
        !$OMP PARALLEL DO PRIVATE(n, Er)
            do n=1,Nest2
                ei=eimin + n*estp 
                call Er_bisection(ei, mi, ResEl, ResEr, ni, Ern(n), denstas(n))
            end do
         !$OMP END PARALLEL DO
            max_pos = maxloc(denstas, dim = 1)
            ei= eimin+max_pos*estp
            eimin=ei-2.d0*estp
            estp=estp/10
        end do
        er = Ern(max_pos)
        dens= denstas(max_pos)   
        deallocate(denstas, Ern)
        free = abs(CLD0(GrF%it,GrF%kappa,er,ei))

        if(dens/free.gt.10)write(6,'(2a10,I2,2f12.6,2e12.5,f5.1)')lev(it)%tb(i0),'Erni=',ni,er,ei,dens,free,dens/free
        if (dens/free.gt.10.and.dens.gt.5) then
            call Check_density(er,ei,dens,C)
            return
        end if
        if (dens==0.d0) Erlpr=.false.


End subroutine Er_Reson_state
subroutine GF_bound_state(eig,err)
    integer:: it,ka,i0
    double precision, intent(in) :: eig
    double precision, intent(out) :: err
    integer :: n, Nest, max_pos, Nest2
    double precision :: epsne, estpne, emaxr, emaxl, er, ei, C, eigg, dens
    double precision, allocatable :: denstas(:)
    ei=1.0d-6  ! smoothing parameter
    estpne=1.d-3   
    it=GrF%it; ka=GrF%kappa;  i0=GrF%i0
    eigg=my_round(eig, 3)
    emaxl=eigg-0.15; emaxr=eigg+0.15; if(emaxr.gt.Cst) emaxr=Cst
    Nest=(emaxr-emaxl)/estpne+1;    Nest2 = Nest/2
    allocate(denstas(0:Nest))
        do while (estpne.ge.er_preci)
           denstas=0.d0
           !$OMP PARALLEL DO PRIVATE(n, Er)
           do n=0,Nest
               Er=emaxl + n*estpne 
               denstas(n)=CLD(it,ka,er,ei)
           end do
           !$OMP END PARALLEL DO
           max_pos = maxloc(denstas, dim = 1)-1
           Er=emaxl + max_pos*estpne
           estpne= estpne/10   
           emaxl = Er-Nest2*estpne
        end do
        dens = denstas(max_pos)
    deallocate(denstas)
        if (dens.gt.2 .and. er.lt.Cst) call Check_density(er,ei,dens,C)
        err=er
        !if (C.lt.0.5.and.eig.lt.Cst)  write(*,'(a,2x,a,f15.8,2x,a)') "Green's Function failed in solving Dirac equation!",lev(it)%tb(i0),eig,'±0.5 MeV'
End subroutine GF_bound_state
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
Subroutine Check_density(er,ei,dens,C)
  
    integer :: it, i0, nc, ka
    double precision, intent(in) :: er, ei, dens
    double precision , intent(out) ::  C
    double precision , dimension(MSD) :: fun
    double precision :: cc, alpha
    integer :: n, Nest, nestg, ni, npt, i
    complex*16 Ec, ui
    it=GrF%it; i0=GrF%i0; ka=GrF%kappa; nc=GrF%node 
    ui=(0.d0, 1.d0);    npt = well%npt 
    ResGF(i0,it)%Hgg = 0.d0; ResGF(i0,it)%Hgf = 0.d0; ResGF(i0,it)%Hff = 0.d0
    if(non_local)then
        ResGF(i0,it)%gg = 0.d0; ResGF(i0,it)%gf = 0.d0; ResGF(i0,it)%ff = 0.d0
    end if
    
    if (ei.lt.0.01) then
        call Res_GF(it,ka,er,ei)
        
            fun(1:npt) = real(GrF%Res_Hgg(1:npt)) + real(GrF%Res_Hff(1:npt))
            call simps(fun, npt, well%h, C)
            ResGF(i0,it)%Hgg = real(GrF%Res_Hgg)/C
            lev(it)%N(i0)  = CountLocalMaxima(real(GrF%Res_Hgg), npt, well%h, ei) 
            if(nc.ne.lev(it)%N(i0)) then
                write(*,'(2a10,2x,f12.6,I5,I5)')lev(it)%tb(i0),'C,node',C,lev(it)%N(i0),nc
                return    
            end if
            ResGF(i0,it)%Hgf = real(GrF%Res_Hgf)/C;  ResGF(i0,it)%Hff = real(GrF%Res_Hff)/C
            if(non_local)then
               ResGF(i0,it)%gg = real(GrF%Res_gg)/C;       ResGF(i0,it)%gf = real(GrF%Res_gf)/C
               ResGF(i0,it)%ff = real(GrF%Res_ff)/C
            end if
    else
        Ec = er-ui*0.d0; C=1.d0
        call DiracGF(it,ka,Ec,non_local)
        fun = -aimag(DOS%HGG)-aimag(DOS%HFF)
        call simps(fun,npt,well%h, C)
        
        ResGF(i0,it)%Hgg=-aimag(DOS%HGG)/C;     ResGF(i0,it)%Hff=-aimag(DOS%HFF)/C;     
        ResGF(i0,it)%Hgf=-aimag(GrF%Hgf)/C

        lev(it)%N(i0)  = CountLocalMaxima(ResGF(i0,it)%Hgg, npt, well%h, ei)
        if(200.eq.lev(it)%N(i0))  return 
        if(non_local)then
            ResGF(i0,it)%gg=-aimag(GrF%Fgg)/C;  ResGF(i0,it)%ff=-aimag(GrF%Fff)/C
            ResGF(i0,it)%gf=-aimag(GrF%Fgf)/C     
        end if
    end if
    lev(it)%ee(i0) = er; lev(it)%hgam(i0) = ei;  lev(it)%lpG(i0) =.true.
    lev(it)%CC(i0) = C ; lev(it)%den(i0) = dens
End Subroutine Check_density
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
! !                                                                                                 !
! Function denstatesVS0(it,ka,er,ei) 
    
!     double precision, intent(in) :: ei, er
!     double precision :: sum1, h, denstatesVS0, fun(MSD)
!     integer, intent(in) :: ka, it
!     integer :: i
!     complex*16 :: ui, Ec
!     ui=(0.d0, 1.d0) 
!     Ec = er-ui*ei
!     call DiracGFVS0(it,ka,Ec,.false.)
!     fun=aimag(DOS%HGG+DOS%HFF)
!     call simps(fun, well%npt, well%h, sum1)
!     denstatesVS0=sum1/pi   !*2*abs(ka)
! !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
! !                                                                                                 !
! End function denstatesVS0                                                                         !
! !                                                                                                 !
! !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
! Function denstates(it,ka,er,ei) 
!     double precision, intent(in) :: ei,er
!     double precision :: h, ff7, sum1, denstates, fun(MSD)
!     integer, intent(in) :: ka, it
!     integer :: i
!     complex*16 :: ui, Ec
!     ui=(0.d0, 1.d0)
!     denstates=0.d0  
!     Ec = er-ui*ei
!     call DiracGF(it,ka,Ec,.false.)
!     fun=aimag(DOS%HGG+DOS%HFF)
!     call simps(fun, well%npt, well%h, sum1)
!     denstates=sum1/pi   !*2*abs(ka)
! !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
! !                                                                                                 !
! End function denstates                                                                               !
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
integer function count_leading_zeros(x) result(zeros)
double precision, intent(in) :: x
character(len=15) :: str
integer :: point_pos, i, str_len

zeros = 6  

write(str, '(F15.10)') x  ! 30位宽度，保留10位小数
str = adjustl(str)        ! 左对齐去除前导空格

! 查找小数点位置
point_pos = index(str, '.')

! 截取小数部分
str = str(point_pos+1:)    ! 保留小数点后的内容
str_len = len_trim(str)    ! 获取有效长度

! 搜索第一个非零字符
do i = 1, str_len
    if (str(i:i) /= '0') then
        zeros = i      ! 有效位数
        return
    end if
end do
! 如果没有非零字符，返回字符串长度
end function count_leading_zeros

subroutine bisection(er, mi, ni, ei, fr) 
    implicit none
    integer, intent(in) :: ni, mi
    double precision, intent(in) :: er
    double precision, intent(out) :: ei, fr
    integer :: it, ka, n, nmax, i, point_pos
    double precision :: left, right, mid, error, prec, fm, fl, DeltaE, hmin
    character(len=15) :: str
    it=GrF%it;  ka=GrF%kappa; hmin=1.d-11
    
    prec = 1.0D0 / (10.0D0 ** mi); error=1.d-4; right = hgam_max  ; DeltaE=0.1
    if (mi.eq.count_leading_zeros(er_preci)-1) prec = error
    fr=CLD(it, ka, er, prec)
    if (ni==1.and.fr.gt.0.d0) then
        if (prec.gt.error) then
            ei=prec
            return
        else
            fl=CLD(it, ka, er, hmin)
            ! Gama/2 < 1.d-12
            if(fl.gt.0.d0) THEN
                ei=hmin
                fr=fl
                return
            else
                left  = hmin
                right = error
            end if
        ! 1.d-5 < Gama/2 < 1.d-10
            do while (abs(right - left) > 0.1*hmin)
                mid = (left+right)/2.d0
                fm= CLD(it, ka, er, mid)!-CLD0(it,ka,er,mid)
                if ( fm*fl.le. 0.0) then
                    right = mid
                    fr = fm
                elseif ( fm*fr .lt. 0.0 ) then
                    left = mid
                    fl= fm
                end if
            end do
            fr=abs(fr) ;    fl=abs(fl)
            if (fl.le.fr) then
                ei = right
            else
                ei = left
                fr = fl
            end if
            return
        end if
    end if
    nmax = right/DeltaE+1
        
    i=1
    do n=1, nmax
        right=DeltaE*n
        fl=fr
        fr= CLD(it, ka, er, right)
        if (fl*fr.lt.0.0) then
            if (ni==i) exit
                i=i+1
        end if
    end do
    left=right-DeltaE
    if (fl*fr.gt.0.0.or.ni.gt.i) then
        ei=0.d0
        fr=0.d0
        return
    end if
    do while (abs(right - left) > 0.1*prec)
        mid = my_round((left+right)/2.d0, mi+1)
        fm= CLD(it, ka, er, mid)
        if ( fm*fl.le. 0.0) then
            right = mid
            fr = fm
        else
            left = mid
            fl= fm
        end if
    end do

    fr=abs(fr) ;    fl=abs(fl)
    if (fl.le.fr) then
        ei = right
    else
        ei = left
        fr = fl
    end if
   ! if (ka.eq.-2)write(*,'(a,2x,3f12.6)')'Bisection:',er,ei, fr
    ! pause
end subroutine bisection
subroutine Er_bisection( ei, mi, ermin, ermax, ni, er, fr) 
    implicit none
    integer, intent(in) :: ni, mi
    double precision, intent(in) :: ei, ermin, ermax
    double precision, intent(out) :: er, fr
    integer ::  it, ka, n, nmax, i
    double precision :: left, right, mid, error, fm, fl, DeltaE
    it=GrF%it;  ka=GrF%kappa
    error=1.0D0 / (10.0D0 ** mi)
    left = ermin; right = ermax  ; DeltaE=0.05
    fr=CLD(it, ka, left, ei)
    nmax = (ermax-ermin)/DeltaE+1;  i=1
    do n=1, nmax
        right=ermin+DeltaE*n
        fl=fr
        fr= CLD(it, ka, right, ei)
        if (fl*fr.lt.0.0) then
            left=right-DeltaE
            if (ni==i) exit
                i=i+1
        end if
    end do

    if (fl*fr.gt.0.0.or.ni.gt.i) then
        er=0.d0
        fr=0.d0
       ! pause
        return
    end if
    do while (abs(right - left) > 0.2*error)
        mid = my_round((left+right)/2.d0, mi+1)
        fm= CLD(it, ka, mid, ei)
        if ( fm*fl.le. 0.0) then
            right = mid
            fr = fm
        elseif ( fm*fr .lt. 0.0 ) then
            left = mid
            fl= fm
        end if
    end do

    fr=abs(fr) ;    fl=abs(fl)
    if (fl.le.fr) then
        er = right
    else
        er = left
        fr = fl
    end if
    !write(*,'(a,2x,3f12.6)')lev(it)%tb(GrF%i0),er,ei,fr
end subroutine Er_bisection
function my_round(x, n) result(rounded_x)
    double precision, intent(in) :: x
    integer, intent(in) :: n
    double precision :: factor, rounded_x
    factor = 10.0**n
    rounded_x = nint(x * factor) / factor
end function my_round
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
subroutine Res_GF(it,ka,er,hgama)                                                                           !
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
Integer , intent(in) :: it, ka
double precision , intent(in) :: er, hgama
Integer :: nist, nest, n
complex*16 :: ui, Ec, ff, left_up, left_down, right_up, right_down
double precision :: estp, gama, emaxl, emaxr

    ui   = (0.d0, 1.d0);    estp = 0.0001 !energy step unit: MeV
    GrF%Res_gg = (0.d0, 0.d0);   GrF%Res_gf = (0.d0, 0.d0);   GrF%Res_ff = (0.d0, 0.d0)  
    GrF%Res_Hgg = (0.d0, 0.d0);  GrF%Res_Hgf = (0.d0, 0.d0);  GrF%Res_Hff = (0.d0, 0.d0)
  
    nest = 20;       nist=15 
    
    emaxl=er-estp*nest/2.d0;   emaxr=er+estp*nist/2.d0;

    left_up   = emaxl + ui*( estp*nist/2.d0-hgama);     Right_up   = emaxl + estp*nest + ui*( estp*nist/2.d0-hgama)
    left_down = emaxl + ui*(-estp*nist/2.d0-hgama);     Right_down = emaxl + estp*nest + ui*(-estp*nist/2.d0-hgama)

   ! write(*,'(a,2x,8f12.6)')'left_up,Right_up,left_down,Right_down=',left_up,Right_up,left_down,Right_down
!!!!$omp parallel do private(ff2, Ec) reduction(+:GrF%Res_gg, GrF%Res_ff, GrF%Res_gf, GFn11, GFn22, GFn12, GFn21)
    do n=0, nest
        if (n.eq.0 .or. n.eq.nest ) then
            ff=0.5d0*estp/(2.0d0*pi*ui)
        else
            ff=estp/(2.0d0*pi*ui)
        end if
!right_down ----> left_down
        Ec = Right_down - estp*n
        call DiracGF(it,ka,Ec,non_local)
        call GF_SUM(ff)
        ! call DiracGFVS0(it,ka,Ec,non_local)
        ! call GF_SUM(-ff)
!left_up -----> right_up
        Ec = left_up + estp*n
        call DiracGF(it,ka,Ec,non_local)
        call GF_SUM(-ff)
        ! call DiracGFVS0(it,ka,Ec,non_local)
        ! call GF_SUM(ff)
    enddo !end for E contour path
!!!$omp end parallel do
!!!$omp parallel do private(ff2, Ec) reduction(+:GrF%Res_gg, GrF%Res_ff, GrF%Res_gf, GFn11, GFn22, GFn12, GFn21)
    do n=0, nist
        if (n.eq.0 .or. n.eq.nist) then
            ff=0.5d0*ui*estp/(2.0d0*pi*ui)
        else
            ff=ui*estp/(2.0d0*pi*ui)
        end if
! right_up -----> right_down
        Ec=Right_up - ui*estp*n
        call DiracGF(it,ka,Ec,non_local)   
        call GF_SUM(ff)
        ! call DiracGFVS0(it,ka,Ec,non_local)
        ! call GF_SUM(-ff)
            
! left_down -----> left_up
        Ec=left_down + ui*estp*n
        call DiracGF(it,ka,Ec,non_local)
        call GF_SUM(-ff)
        ! call DiracGFVS0(it,ka,Ec,non_local)
        ! call GF_SUM(ff)
    enddo !end for E contour path
!!!$omp end parallel do       
    GrF%Res_Hgg(1)  = 3.*(GrF%Res_Hgg(2) - GrF%Res_Hgg(3)) + GrF%Res_Hgg(4)
    GrF%Res_Hff(1)  = 3.*(GrF%Res_Hff(2) - GrF%Res_Hff(3)) + GrF%Res_Hff(4)
End Subroutine Res_GF  
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
subroutine GF_SUM(ff2)                                                                           !
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
    complex*16 , intent(in) :: ff2 

    GrF%Res_Hgg = GrF%Res_Hgg + ff2*DOS%HGG
    GrF%Res_Hff = GrF%Res_Hff + ff2*DOS%HFF
    GrF%Res_Hgf = GrF%Res_Hgf + ff2*GrF%Hgf

    if(non_local) then
        GrF%Res_gg = GrF%Res_gg + ff2*GrF%Fgg
        GrF%Res_ff = GrF%Res_ff + ff2*GrF%Fff
        GrF%Res_gf = GrF%Res_gf + ff2*GrF%Fgf
    end if    
End Subroutine GF_SUM
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
  subroutine DiracGFwav(it,ka,Ec,lpr)                                                                !
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
    logical lpr!,lpe
    double precision :: h, h2, r, r1, r2, r4, original!, a,b,c
    complex*16 ep2amu, akvec, Ec, eigfm, cww, twocww!, cw2, cw3, cw4
    complex*16, dimension(4) :: ag, af
    complex*16 gin(msd),fin(msd),gout(msd),fout(msd)
    complex*16  sg1, sf1, sg2, sf2, u1g, u1f, u2g, u2f, u1x, u2y
    integer :: i0, it, ka, i, jk, npt, m0, j, n, nmax, n0, ib
    complex*16 :: alph, beta, gmesh, fmesh
    double precision, dimension(2) ::  tc
    data tc/0.0d0, 1.0d0/

!--- Initialization of state's quantities(eigfm - dpotl(it)%vps(npt))
    h      = well%h;               h2   = well%h*half;           npt = well%npt
    ib          = chunk(it)%ib(ka)
    n0          = chunk(it)%ia(ib)
    nmax        = chunk(it)%id(ib)
    DOS%HGG = zero;  DOS%Hff = zero;    GrF%Hgf = zero
    do n = 1, nmax
        i0              = n0 + n
        !write(*,*)lev(it)%tb(i0), lev(it)%ee(i0)
        
        DOS%HGG = DOS%HGG + wav(i0,it)%g*wav(i0,it)%g*hbc/(Ec-lev(it)%ee(i0))
        DOS%HFF = DOS%HFF + wav(i0,it)%f*wav(i0,it)%f*hbc/(Ec-lev(it)%ee(i0))
        GrF%Hgf = GrF%Hgf + wav(i0,it)%g*wav(i0,it)%f*hbc/(Ec-lev(it)%ee(i0))
    end do

    if(.not.lpr) return 
    GrF%Fgg=zero; GrF%Fgf=zero;  GrF%Fff=zero
    do n = 1, nmax
        i0              = n0 + n
      do j=1,npt !r'
        do i=1,npt !r
            GrF%Fgg(i,j) = GrF%Fgg(i,j) + wav(i0,it)%g(i)*wav(i0,it)%g(j)*hbc/(Ec-lev(it)%ee(i0))
            GrF%Fff(i,j) = GrF%Fff(i,j) + wav(i0,it)%f(i)*wav(i0,it)%f(j)*hbc/(Ec-lev(it)%ee(i0))
            GrF%Fgf(i,j) = GrF%Fgf(i,j) + wav(i0,it)%g(i)*wav(i0,it)%f(j)*hbc/(Ec-lev(it)%ee(i0))
        end do
      end do
    end do
    
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
    End subroutine DiracGFwav
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
  subroutine DiracGF(it,ka,Ec,lpr)                                                                !
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
    logical lpr!,lpe
    double precision :: h, h2, r, r1, r2, r4, original!, a,b,c
    complex*16 ep2amu, akvec, Ec, eigfm, cww, twocww!, cw2, cw3, cw4
    complex*16, dimension(4) :: ag, af
    complex*16 gin(msd),fin(msd),gout(msd),fout(msd)
    complex*16  sg1, sf1, sg2, sf2, u1g, u1f, u2g, u2f, u1x, u2y
    integer :: i0, it, ka, i, jk, npt, m0, j, n
    complex*16 :: alph, beta, gmesh, fmesh
    double precision, dimension(2) ::  tc
    data tc/0.0d0, 1.0d0/

!--- Initialization of state's quantities(eigfm - dpotl(it)%vps(npt))
    h      = well%h;               h2   = well%h*half;           npt = well%npt;         eigfm   = Ec/hbc
     ! ep2amu = eigfm + emcc(it);    akvec = eigfm*ep2amu
      ep2amu = eigfm -dpotl(it)%vms(npt);     akvec = (eigfm-dpotl(it)%vps(npt))*ep2amu
!--- Determine the wave functions on the original and infinity position
    gin = (0.d0,0.d0);                     fin  = (0.d0,0.d0)
    gout = (0.d0,0.d0);                     fout  = (0.d0,0.d0) 
     
    Call GF_anaori(eigfm, ka, it, h, gmesh, fmesh)
    gin(2) = gmesh ;                    fin(2) = fmesh 
       !  c----boundary condition at r-->Infinity
    call anainf2(eigfm, akvec, ep2amu, ka, gout(npt), fout(npt))
!--- check with box boundary ---
                ! gout(npt)=0.d0
                ! fout(npt)=1.0d0

    call Runge_Kutta(it,ka,eigfm,gin,fin,gout,fout)

    !if(IE.eq.2) call SolveDirac(it,ka,eigfm,Gin,Fin,gout,fout)

    m0=npt-50
    cww=(gin(m0)*fout(m0)-gout(m0)*fin(m0))*hbc
    twocww=2.0d0*cww
    DOS%HGG=(gout*gin+gin*gout)/twocww
    GrF%Hgf=(gout*fin+gin*fout)/twocww
    DOS%HFF=(fout*fin+fin*fout)/twocww

    if(.not.lpr) return 
      do j=1,npt !r'
        do i=1,npt !r
            if (i.gt.j)then !r>r'
                GrF%Fgg(i,j)=gout(i)*gin(j)/cww
                GrF%Fgf(i,j)=gout(i)*fin(j)/cww
                GrF%Fff(i,j)=fout(i)*fin(j)/cww
            elseif(i.lt.j)then !r<r'
                GrF%Fgg(i,j)=gin(i)*gout(j)/cww
                GrF%Fgf(i,j)=gin(i)*fout(j)/cww
                GrF%Fff(i,j)=fin(i)*fout(j)/cww
            elseif(i.eq.j)then !r=r'
                GrF%Fgg(i,j)=(gout(i)*gin(i)+gin(i)*gout(i))/twocww!DOS%HGG(i)
                GrF%Fgf(i,j)=GrF%Hgf(i)
                GrF%Fff(i,j)=(fout(i)*fin(i)+fin(i)*fout(i))/twocww!DOS%HFF(i)
            end if
        end do
      end do

    
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
    End subroutine DiracGF
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
  subroutine DiracGFVS0(it,ka,Ec,lpr)                                                                              !
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
    logical lpr
    double precision :: h, h2, r, r1, r2, r4, original!, Vcou(MSD),Vcouh(MSD)!,vms(MSD),vmsh(MSD)
    complex*16 ep2amu, akvec, Ec, eigfm, cww, twocww!, cw2, cw3, cw4
    complex*16, dimension(4) :: ag, af
    complex*16 gin(msd),fin(msd),gout(msd),fout(msd)
    complex*16  sg1, sf1, sg2, sf2, u1g, u1f, u2g, u2f, u1x, u2y
    integer :: i0, it, ka, i, jk, npt, m0, j
    complex*16 :: alph, beta, gmesh, fmesh
    double precision, dimension(2) ::  tc
    data tc/0.0d0, 1.0d0/
    if (real(Ec).le.0.d0) return
!--- Initialization of state's quantities
    h      = well%h;               h2    = well%h*half;        npt = well%npt;         eigfm   = Ec/hbc
    ! ep2amu = eigfm + emcc(it);    akvec = eigfm*ep2amu
    ep2amu = eigfm -dpotl(it)%vms(npt);     akvec = (eigfm-dpotl(it)%vps(npt))*ep2amu
    

   
    gin  = (0.d0,0.d0);                     fin  = (0.d0,0.d0)
    gout = (0.d0,0.d0);                     fout = (0.d0,0.d0)
    if(ka.gt.0) then
        original=h**(ka+1)
        alph    = ((two*ka + one)/h)*original !+ dpotl(it)%vt(2) + XG(2))*original 
        beta    = eigfm - dpotl(it)%vms(2) !- XF(2)
        gmesh   = original
        fmesh   = alph/beta
    else
        original=h**(-1*ka) 
        alph    = (eigfm - tc(it)*dself%cou(2))*original!-dpotl(it)%vps(2) - YG(2))*original 
        beta    = (two*ka - 1)/h!  + dpotl(it)%vt(2) + YF(2)
        gmesh   = original
        fmesh   = alph/beta
    end if 
    gin(2) = gmesh ;                    fin(2) = fmesh  
    
!  c----boundary condition at r-->Infinity
    call anainf2(eigfm, akvec, ep2amu, ka, gout(npt), fout(npt))

    call Runge_Kutta_Free(it, ka, eigfm, gin, fin, gout, fout)
    m0=npt-50
    cww=(gin(m0)*fout(m0)-gout(m0)*fin(m0))*hbc
    twocww=2.0d0*cww
    DOS%HGG=(gout*gin+gin*gout)/twocww
    DOS%HFF=(fout*fin+fin*fout)/twocww
    GrF%Hgf=(gout*fin+gin*fout)/twocww
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
    End subroutine DiracGFVS0
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!  

  FUNCTION CountLocalMaxima(GG, n, h, ei) RESULT(count)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: n
    double precision , INTENT(IN) :: GG(n), ei
    INTEGER :: count, i, j, window_size = 10
    double precision :: h, min_val, max_val
    
    count = 0.
        DO i = 11, n-2
         !IF (GG(i) < -1.d-5) return
            IF (GG(i) > GG(i-1) .AND. GG(i) > GG(i+1)) THEN
                IF (GG(i) .gt. 0.01) THEN
                    count = count + 1
                END IF
            END IF
        END DO
        if (maxval(GG).eq.GG(n)) count=90+count
    !  end if
109 format(f8.3, 7E17.8)
END FUNCTION CountLocalMaxima

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
Subroutine GFDetXY(i0, it, lpr)                                                                          !
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!--- Calculate the X and Y Components, and determine equivalent localized XG, XF, YG, and YF terms
!
    integer, intent(in) :: i0, it
    integer :: i, j, n, ib, ka
    logical, intent(in) :: lpr
    complex*16, dimension(MSD) :: S
    if (.not. lev(it)%lpG(i0)) then
         call DetXY(i0,it)
         return
     end if
    n   = well%npt
    ka  = lev(it)%nk(i0)
    ib  = chunk(it)%ib(ka)
!  write(*,'(a, 4i3)')'XY',ka,ib,i0,it
    XG = 0.d0; XF = 0.d0; YG = 0.d0; YF = 0.d0
!$OMP PARALLEL DO private(i) SCHEDULE(STATIC)
    do j = 2, n !r      
        do i = 2, n !r'
          XG(j) = XG(j) + epotl(ib,it)%XG(j,i)*ResGF(i0,it)%gg(j,i) + epotl(ib,it)%XF(j,i)*ResGF(i0,it)%gf(j,i)
          XF(j) = XF(j) + epotl(ib,it)%XG(j,i)*ResGF(i0,it)%gf(i,j) + epotl(ib,it)%XF(j,i)*ResGF(i0,it)%ff(j,i)
          YG(j) = YG(j) + epotl(ib,it)%YG(j,i)*ResGF(i0,it)%gg(j,i) + epotl(ib,it)%YF(j,i)*ResGF(i0,it)%gf(j,i)
          YF(j) = YF(j) + epotl(ib,it)%YG(j,i)*ResGF(i0,it)%gf(i,j) + epotl(ib,it)%YF(j,i)*ResGF(i0,it)%ff(j,i)   
        end do
    end do
!$OMP END PARALLEL DO 
    S(2:n)  = one/( ResGF(i0,it)%Hgg(2:n) + ResGF(i0,it)%Hff(2:n) )

        XG(2:n)   = XG(2:n) * S(2:n)
        XF(2:n)   = XF(2:n) * S(2:n)
        YG(2:n)   = YG(2:n) * S(2:n)
        YF(2:n)   = YF(2:n) * S(2:n)
    XG(1) = 3.0*(XG(2) - XG(3)) + XG(4); XF(1)=3.0*(XF(2) - XF(3)) + XF(4)
    YG(1) = 3.0*(YG(2) - YG(3)) + YG(4); YF(1)=3.0*(YF(2) - YF(3)) + YF(4)

    call intpol6(XG, XGh, n);               Call intpol6(XF, XFh, n)
    call intpol6(YG, YGh, n);               Call intpol6(YF, YFh, n)


    ! if (lpr) then
    !     open(21, file = 'XY/'//name(1:IFN)//lev(it)%tb(i0)(1:6)//'.'//pset%txtfor, status = 'unknown')
    !     write(21,'(a8,11a20)') lev(it)%tb(i0),'Y_G','Y_F','X_G','X_F','YG','XF',pset%txtfor
    !     do i = 1, n
    !         write(21,'(f8.4,11Es20.8)') well%xr(i), YG(i)*hbc, YF(i)*hbc, XG(i)*hbc, XF(i)*hbc
    !     end do
    !     close(21)
    ! end if
    return

! !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
End Subroutine GFDetXY                                                                              !
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!

Subroutine anainf2(ee, kk,ep2amu,kap,ginf,finf)
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
    complex*16, intent(in) :: kk, ep2amu
    complex*16, intent(out) :: ginf, finf
    integer, intent(in) :: kap
    integer :: npt
    double precision ::  rmax
    complex*16 :: akvec, akrmax
    complex*16:: csphhf, akk, ee, ui, eta, etiphi, phi, coup
    double precision :: Rekvec, Imkvec, gamma, phase, m
    ui=(0.d0, 1.d0)
!--- This subroutine is used to calculate the asymptotic behavior of wave functions at infinity
!--- ginf and finf are the values of large and small components at infinity  
    npt     = well%npt;       rmax = well%xr(npt)
    akvec  = sqrt(kk)
    Rekvec = abs(real(akvec))
    Imkvec = abs(aimag(akvec))
    if (aimag(kk).gt.0.d0) then
        akvec = cmplx(Rekvec,Imkvec)
    else
        akvec = cmplx(Rekvec,-Imkvec)
        if (real(ee).lt.Cst/hbc) akvec = -akvec
    end if
    phase = one
    ! coup = 0.d0
    ! if (GrF%it.eq.2) then
    !     gamma = sqrt(kap**2-(nuc%npr(2)/alphi)**2)
    !     phase = rmax**(gamma+GrF%node-1)
    ! !     m=emcc(2)/2.
    ! !     eta = nuc%npr(2)*(ee+m)/(alphi*akvec)
    ! !     etiphi = -(kap+ui*eta*m/(ee+m))/(gamma-ui*eta)
    ! !     phi = 0.d0!log(etiphi)/(2.d0*ui)
    ! ! coup=phi-eta*log(2.d0*akvec*rmax)-pi*gamma/2.d0-arg_gamma(gamma, eta)
    ! !     phase = exp(ui*coup)
    ! end if
    ! TTSun
    ! if(ee.lt.0.d0) then
    !   akvec = cmplx(real(akvec),abs(aimag(akvec)))
    ! else
    !   akvec = cmplx(real(akvec),-abs(aimag(akvec)))
    ! endif

     akrmax = akvec*rmax
     akk    = akrmax/ep2amu
     if(kap.gt.0.d0) then
       call spherical_hankel1(kap-1,akrmax,csphhf)
       finf = akk*csphhf*phase
       call spherical_hankel1(kap,akrmax,csphhf)
       ginf = rmax*csphhf*phase
     else
       call spherical_hankel1(-kap,akrmax,csphhf)
       finf = -akk*csphhf*phase
       call spherical_hankel1(-kap-1,akrmax, csphhf)
       ginf = rmax*csphhf*phase
     endif

    return

    End Subroutine anainf2
 
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
Subroutine GF_anaori(eigfm, ka, it, h, gmesh, fmesh)                                             !
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!--- This subroutine is used to the original behavior of wave functions
!
    double precision, intent(in) :: h
    complex*16, intent(in) :: eigfm
    integer, intent(in) :: ka, it
    complex*16 , intent(out) :: gmesh, fmesh
    complex*16  :: alph, beta, original

    if(ka.gt.0) then
        original=h**(ka+1)
        alph    = ((two*ka + one)/h + dpotl(it)%vt(2) + XG(2))*original 
        beta    = eigfm - dpotl(it)%vms(2) - XF(2)
        gmesh   = original
        fmesh   = alph/beta
    else
        original=h**(-1*ka) 
        alph    = (eigfm - dpotl(it)%vps(2) - YG(2))*original 
        beta    = (two*ka - 1)/h  + dpotl(it)%vt(2) + YF(2)
        gmesh   = original
        fmesh   = alph/beta
    end if 
    return
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
End subroutine GF_anaori                                                                             !
!                                                                                                 !
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!                                                                                                 !
Subroutine SPE_DOS                                                                                !
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!
    logical :: lprx
    integer :: ib, i0, n0, nmax, n, it, i, ip, Nest, Nest1
    integer :: ka, nc, max_pos, nn, j
    double precision :: er, ei, ResEl,ResEr, DeltaE, nt, free, emaxl, estpne
    double precision :: C, estp, E_min, E_max, prec
    double precision, allocatable :: denstas(:)
    Call system("mkdir -p DOS")
    er_step=0.1
    Nest=(ecut+10.)/er_step
    allocate(denstas(0:Nest))
    XF=zero ;   XG=zero;  YF=zero;     YG=zero
    XFh=zero;   XGh=zero;   YFh=zero;   YGh=zero
!--- Loop over neutron and proton
    do it = 1, 2
    if (.not.lprNP(it)) cycle
    !--- Loop over the single particle levels
    do ib = 1, chunk(it)%nb
        n0          = chunk(it)%ia(ib)
        nmax        = chunk(it)%id(ib)
        ka          = chunk(it)%kbl(ib) 
        do nc = 1, nmax
            i0              = n0 + nc
            if(lev(it)%ee(i0).lt.0.d0) cycle
            prec = lev(it)%ee(i0) - aint(lev(it)%ee(i0) * 10.0d0 + 1d-12) / 10.0d0
            !--- Calculate XG, XF, YG, YF
            if(IE.eq.2) call GFDetXY(i0,it,.false.)   !--- Calculate XG, XF, YG, YF
            if (SPE.gt.0) then
                open(44, file = 'DOS/'//name(1:IFN)//tex%tit(it)//'.'//lev(it)%tb(i0)(4:6)//'.2.'//pset%txtfor, status = 'unknown')
                write(44, '(2a12,4a20)') 'Er_'//pset%txtfor,'Nt_'//pset%txtfor,'Nf_'//pset%txtfor,'Res_'//pset%txtfor,lev(it)%tb(i0)
            end if
            if (SPE==2) then
                write(44,'(f12.6,Es20.8,a20,Es20.8)') -70.d0, 0.d0, '---', 0.d0
                if (chunk(it)%ia(ib)+1.lt.i0) then
                    do i=chunk(it)%ia(ib)+1, i0-1
                        write(44,'(f12.6,Es20.8,a20,Es20.8)')lev(it)%ee(i),0.d0,'---',0.d0
                        write(44,'(f12.6,Es20.8,a20,Es20.8)')lev(it)%ee(i),lev(it)%den(i),'---', lev(it)%den(i)
                        write(44,'(f12.6,Es20.8,a20,Es20.8)')lev(it)%ee(i),0.d0,'---',0.d0
                    end do
                end if
            end if
            denstas=0.d0
            Er = 1.d-6; i=0
            nt = -2.d0*abs(ka)*CLD(it,ka,er,1.d-6)/pi
            free = -2.d0*abs(ka)*CLD0(it,ka,er,1.d-6)/pi
            denstas(0) = nt - free
            if(SPE.gt.0) write(44,'(f12.6,3Es20.8)')er,nt,free,denstas(0)
            do n=1,Nest
                Er = n*er_step + prec
                nt = -2.d0*abs(ka)*CLD(it,ka,er,1.d-6)/pi
                free = -2.d0*abs(ka)*CLD0(it,ka,er,1.d-6)/pi
                denstas(n) = nt - free
                if(SPE.gt.0) write(44,'(f12.6,3Es20.8)')er,nt,free,denstas(n)
                if (denstas(n-1).gt.1.d-4)THEN
                    if (denstas(n).le.1.d-4 .or. n.eq.Nest)THEN
                        E_max=Er + er_step 
                        goto 20
                    end if
                endif
               cycle     
              
        20  if (E_max.ge.lev(it)%ee(i0))  THEN
                i0=i0+1!; nc=nc+1
                if (nc.gt.chunk(it)%id(ib)) exit
                    if(IE.eq.2) call GFDetXY(i0,it,.false.)
                end if
                denstas=0.d0
                prec = lev(it)%ee(i0) - aint(lev(it)%ee(i0) * 100.0d0 + 1d-12) / 100.0d0
            end do
            
            if(SPE.gt.0) close(44)
            exit
        end do !nc
    end do !kap
end do!it
    deallocate(denstas)
    return
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
End Subroutine SPE_DOS                                                                          !
!                                                                                                 !

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
Function CLD0(it,ka,er,ei) 
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!    
    double precision, intent(in) :: ei, er
    double precision :: CLD0, fun(MSD)
    integer, intent(in) :: ka, it
    integer :: i
    complex*16 :: ui, Ec
    ui=(0.d0, 1.d0) 
    Ec = er-ui*ei
    call DiracGFVS0(it,ka,Ec,.false.)
    fun=aimag(DOS%HGG+DOS%HFF)
    call simps(fun, well%npt, well%h, CLD0)
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
End function CLD0                                                                                 !
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
Function CLD(it,ka,er,ei) 
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!    
    double precision, intent(in) :: ei, er
    double precision :: CLD, fun(MSD)
    integer, intent(in) :: ka, it
    integer :: i
    complex*16 :: ui, Ec
    ui=(0.d0, 1.d0) 
    Ec = er-ui*ei
    call DiracGF(it,ka,Ec,.false.)
    fun=aimag(DOS%HGG+DOS%HFF)
    call simps(fun, well%npt, well%h, CLD)
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!                                                                                                 !
End function CLD                                                                                 !
!                                                                                                 !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
subroutine gh_legendre(lmax, g, h)
!------------------------------------------------------------
! 计算 g(theta) 和 h(theta)
!      g = P_l(cosθ)
!      h = sinθ * dP_l(cosθ)/d(cosθ)
!
! 输入:
!   theta  角度 (弧度)
!   l      勒让德多项式的阶数 (l >= 0)
! 输出:
!   g, h   如上述公式
!------------------------------------------------------------
  implicit none
  integer,  intent(in)  :: lmax
  double precision, intent(out) :: g(0:10, 0:60),h(0:10, 0:60)
  double precision :: theta, x, s
  double precision :: p0, p1, p2         ! 用于递推的 P_{l-2}, P_{l-1}, P_l
  integer  :: j, l
  double precision, parameter :: tiny = 1.d-15

  ! 初始化
  g = 0.0;  h = 0.0

  ! l = 0 的情况（对所有角度）
  g(0,:) = 1.0; h(0,:) = 0.0

  ! 角度循环：0° 到 360°，步长 3°
  do j = 0, 60
     theta = j * pi * 3.d0 / 180.
     x = cos(theta)
     s = sin(theta)

     ! --- 先处理 l = 1 ---
     p0 = 1.0          ! P_0
     p1 = x               ! P_1
     g(1,j) = p1
     if (abs(s) < tiny) then
        h(1,j) = 0.d0
     else
        h(1,j) = -1.d0 * (x * p1 - p0) / s
     end if

     ! --- 从 l = 2 递推至 lmax ---
     do l = 2, lmax
        p2 = ((2*l - 1) * x * p1 - (l - 1) * p0) / l   ! 计算 P_l
        g(l,j) = p2
        if (abs(s) < tiny) then
           h(l,j) = 0.d0
        else
           ! 需要 P_{l-1} (即 p1)
           h(l,j) = -l * (x * p2 - p1) / s
        end if
        ! 更新 p0, p1，为下一个 l 准备
        p0 = p1
        p1 = p2
        
     end do !l
     write(*,'(i4, 30f20.8)') 3*j,(g(l,j)**2,l=0,lmax),(h(l,j)**2,l=0,lmax)
  end do! 度
  
end subroutine gh_legendre
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
! subroutine gh_legendre(lmax, g, h)
! !------------------------------------------------------------
! ! 使用 SLATEC 扩展精度库 DXLEGF 精确计算：
! !      g(l, j)   = P_l(cosθ_j)
! !      h(l, j)   = sinθ_j * dP_l(cosθ_j)/d(cosθ_j) = -P_l^1(cosθ_j)
! !
! ! 输入:
! !   lmax  勒让德多项式的最高阶数 (0 <= lmax <= 10)
! ! 输出:
! !   g, h  维度 (0:10, 0:60) 的二维数组，分别存储 g 和 h 值
! !         第一维为阶数 l，第二维为角度序号 j (对应 0°,3°,6°,...,180°)
! !------------------------------------------------------------
!   use Legendre, only: DXLEGF
!   use Legflib, only: DXRED, DXSET
!   implicit none

!   integer, intent(in) :: lmax
!   double precision, intent(out) :: g(0:10, 0:60), h(0:10, 0:60)

!    integer :: nik8, nki8
! !   integer(kind=8) :: mu1, mu2, id, nik8, nki8, ierror
!    INTEGER(kind=8) IERROR, L, MU1, MU2, NUDIFF, J, ID, nuff
!   double precision :: theta, theta1, x, s, pi
!   double precision, allocatable :: pqa(:), pqa1(:)
!   integer(kind=8), allocatable :: ipqa(:), ipqa1(:)
!   double precision :: pval, pval1
!   integer(kind=8) :: ipval, ipval1

 
! !   ! 初始化 DXLEGF 所需的扩展精度环境（如果尚未初始化）
!   ierror = 0
!   nik8 = 0
!   call DXSET(nik8, nik8, 0.0d0, nik8, ierror)
!   if (ierror /= 0) then
!     write(*,*) 'DXSET error: ', ierror
!     return
!   end if

!   ! 临时数组：最多存储 lmax+1 个值
!   allocate(pqa(lmax+1), ipqa(lmax+1))
!   allocate(pqa1(lmax+1), ipqa1(lmax+1))

!   ! 角度循环：0° 到 180°，步长 3°
!   do j = 0, 60
!     theta = j * 3.0d0 * pi / 180.0d0   ! 弧度
!     x = cos(theta)
!     s = sin(theta)

!     ! 特殊情况：theta = 0 或 pi (sinθ = 0)
!     if (abs(s) < 1.0d-14) then
!       do l = 0, lmax
!         g(l, j) = 1.0d0
!         if (mod(l, 2) == 1) g(l, j) = -1.0d0   ! cos(π) = -1 时符号为 (-1)^l
!         if (theta > pi/2 + 1.0d-14) g(l, j) = (-1.0d0)**l   ! 判断是否为 π 附近
!         h(l, j) = 0.0d0
!       end do
!       cycle
!     end if

!     ! 处理 theta <= 90° 和 > 90° 的不同分支
!     if (theta <= pi/2 + 1.0d-14) then
!       ! 直接调用 DXLEGF 计算 P_l(cosθ) 和 P_l^1(cosθ)
!       ! 1) 计算 mu=0 得到 P_l
!       mu1 = 0; mu2 = 0; id = 3          ! ID=3 表示正阶勒让德函数，固定 mu，变化 nu
!       nuff = lmax                       ! NUDIFF = lmax，得到 nu = 0,1,...,lmax
!       call DXLEGF(zero, nuff, mu1, mu2, theta, id, pqa, ipqa, ierror)
!       if (ierror /= 0) then
!         write(*,*) 'DXLEGF error (mu=0) at j=', j, ' ierror=', ierror
!         return
!       end if
!       ! 2) 计算 mu=1 得到 P_l^1
!       mu1 = 1; mu2 = 1
!       call DXLEGF(zero, nuff, mu1, mu2, theta, id, pqa1, ipqa1, ierror)
!       if (ierror /= 0) then
!         write(*,*) 'DXLEGF error (mu=1) at j=', j, ' ierror=', ierror
!         return
!       end if

!       ! 转换扩展精度形式到普通双精度（并规约指数）
!       do l = 0, lmax
!         pval = pqa(l+1); ipval = ipqa(l+1)
!         call DXRED(pval, ipval, ierror)
!         if (ierror /= 0) then
!           write(*,*) 'DXRED error (mu=0) at l=', l, ' ierror=', ierror
!           return
!         end if
!         g(l, j) = pval

!         pval1 = pqa1(l+1); ipval1 = ipqa1(l+1)
!         call DXRED(pval1, ipval1, ierror)
!         if (ierror /= 0) then
!           write(*,*) 'DXRED error (mu=1) at l=', l, ' ierror=', ierror
!           return
!         end if
!         h(l, j) = -pval1   ! h = -P_l^1
!       end do

!     else
!       ! theta > 90°：使用奇偶性转换
!       theta1 = pi - theta   ! 此时 theta1 ∈ (0, π/2)
!       ! 计算 mu=0 和 mu=1 在 theta1 处的值
!       mu1 = 0; mu2 = 0; id = 3
!       nuff = lmax
!       call DXLEGF(zero, nuff, mu1, mu2, theta1, id, pqa, ipqa, ierror)
!       if (ierror /= 0) then
!         write(*,*) 'DXLEGF error (mu=0, theta1) at j=', j, ' ierror=', ierror
!         return
!       end if
!       mu1 = 1; mu2 = 1
!       call DXLEGF(zero, nuff, mu1, mu2, theta1, id, pqa1, ipqa1, ierror)
!       if (ierror /= 0) then
!         write(*,*) 'DXLEGF error (mu=1, theta1) at j=', j, ' ierror=', ierror
!         return
!       end if

!       do l = 0, lmax
!         pval = pqa(l+1); ipval = ipqa(l+1)
!         call DXRED(pval, ipval, ierror)
!         if (ierror /= 0) then
!           write(*,*) 'DXRED error (mu=0, theta1) at l=', l, ' ierror=', ierror
!           return
!         end if
!         ! P_l(cosθ) = (-1)^l * P_l(cosθ1)
!         if (mod(l, 2) == 0) then
!           g(l, j) = pval
!         else
!           g(l, j) = -pval
!         end if

!         pval1 = pqa1(l+1); ipval1 = ipqa1(l+1)
!         call DXRED(pval1, ipval1, ierror)
!         if (ierror /= 0) then
!           write(*,*) 'DXRED error (mu=1, theta1) at l=', l, ' ierror=', ierror
!           return
!         end if
!         ! h = -P_l^1(cosθ) ，而 P_l^1(cosθ) = (-1)^(l+1) P_l^1(cosθ1)
!         ! 所以 h = -[(-1)^(l+1) P_l^1(cosθ1)] = (-1)^l P_l^1(cosθ1)
!         if (mod(l, 2) == 0) then
!           h(l, j) = pval1
!         else
!           h(l, j) = -pval1
!         end if
!       end do
!     end if
!   end do

!   deallocate(pqa, ipqa, pqa1, ipqa1)
! end subroutine gh_legendre
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
subroutine check_ortho(lmax, g)
!------------------------------------------------------------
! 用梯形法则验证勒让德多项式的正交归一性：
! ∫_{-1}^1 P_l(x) P_m(x) dx = 2/(2l+1) δ_{lm}
!
! 输入:
!   lmax    最大阶数
!   g(0:10,0:60)  由 gh_legendre 算出的 P_l(cosθ)
!------------------------------------------------------------
  implicit none
  integer, intent(in) :: lmax
  double precision, intent(in) :: g(0:10, 0:60)

  integer :: j, l, m
  double precision :: theta, s, integral
  double precision, parameter :: pi = 4.d0*datan(1.d0)
  real :: theta_deg

  write(*,*) '========================================'
  write(*,*) ' 验证勒让德多项式正交归一性'
  write(*,*) '   理论值：2/(2l+1)  δ_{lm}'
  write(*,*) '========================================'

  do l = 0, lmax
     do m = 0, lmax
        integral = 0.d0
        ! 梯形法则：步长 3° → dθ = 3*pi/180
        do j = 0, 60
           theta = j * pi * 3.d0 / 180.d0
           s = sin(theta)
           ! 端点权重 0.5，内部点权重 1
           if (j == 0 .or. j == 60) then
              integral = integral + 0.5d0 * g(l,j) * g(m,j) * s
           else
              integral = integral +        g(l,j) * g(m,j) * s
           end if
        end do
        integral = integral * (pi * 3.d0 / 180.d0)   ! 乘以步长

        write(*, '(2i3, f15.10, a, f15.10, a, e12.4)') &
             l, m, integral, '  (理论:', &
             2.d0/(2*l+1) * merge(1.d0, 0.d0, l==m), &
             ')  误差:', &
             integral - 2.d0/(2*l+1) * merge(1.d0, 0.d0, l==m)
     end do
  end do
end subroutine check_ortho
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
Subroutine DCSEl     !Differential scattering cross-section
    integer :: ib, i0, n0, nmax, n, it, i, Nest, Nestj
    integer :: ka, nc, j, lmax, l, ibu, ibd
    integer :: unit, ios, inEnum
    double precision :: er, step, IncE, EcmInc, integral, s, theta
    double precision :: denstas, Sp_t! Scattering phase shift Total, Free, DCSta
    complex*16 :: ui, g(0:60), h(0:60), ss
    double precision :: M, mu, k, kk, fact, p(0:10, 0:60),lp(0:10, 0:60), dcs(0:60)
    double precision, allocatable :: sp(:,:), e(:),Cs(:,:),LD(:,:),den_arr(:)
    double precision, allocatable :: inE(:), kin(:), dcsE(:,:)
    integer, allocatable :: inE_idx(:)
    complex*16 , allocatable ::  hl(:,:), gk(:,:)
    step=1.d-4; j=1000; ui = (0.d0, 1.d0)
    open(newunit=unit, file='dcs.dat', status='old', action='read', iostat=ios)
    if (ios == 0) then! 打开成功，说明文件存在且可读
        read(unit,*) s, Er
        if (ecut.gt.er) ecut = er
        j = int(s/step)  ! 根据文件中的步长调整 j 的值
        read(unit,*) inEnum
        if (inEnum .gt. 0) then
            allocate(inE(inEnum))
            inE=0.d0
            read(unit,*) inE
        end if
        write(*,*) '读取 dcs.dat 文件成功，步长 = ', step, ' j = ', j, ' 输入能量点数 = ', inEnum, ' 输入能量点 = ', inE
        close(unit)
    end if
    Nest=ecut/(step*j); Nestj=Nest*j
    M=nuc%npr(1)*pset%amu(1)+nuc%npr(2)*pset%amu(2)+nuc%etot
    
    allocate(sp(NTX,0:Nestj),e(0:Nestj),cs(NTX+1,0:Nest),LD(NTX,0:Nest))
    e(0)=1.d-6
    do n=1,Nestj
        E(n) = n*step
    end do 
!--- Loop over neutron and proton
    Call system("mkdir -p DCS")
    do it = 1, 2
    if (.not.lprNP(it)) cycle
    
    dcs=0.d0; sp=0.d0; LD=0.d0; cs=0.d0
    mu=M*pset%amu(it)/(M+pset%amu(it))
    EcmInc=(M+pset%amu(it))/M
    !--- Loop over the single particle levels
    do ib = 1, chunk(it)%nb
        n0          = chunk(it)%ia(ib)
        nmax        = chunk(it)%id(ib)
        ka          = chunk(it)%kbl(ib) 
        Sp_t=0.d0
        do nc = 1, nmax
            i0              = n0 + nc
            if(lev(it)%ee(i0).lt.0.d0) then
                Sp_t=Sp_t+pi
                cycle
            end if
            if(IE.eq.2) call GFDetXY(i0,it,.false.)   !--- Calculate XG, XF, YG, YF    
            allocate(den_arr(0:Nestj))        ! 临时数组，存储每个能量点的能级密度差
            den_arr = 0.d0
            !$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(n) SHARED(den_arr, e, it, ka, Nestj)
            do n = 0, Nestj
                den_arr(n) = CLD(it,ka,e(n),1.d-6) - CLD0(it,ka,e(n),1.d-6)
            end do
            !$OMP END PARALLEL DO
            do n=0,Nestj
                Sp_t=Sp_t+den_arr(n)*step;  sp(ib,n) = Sp_t
                if(MOD(n,j).eq.0)then
                    i=n/j
                    LD(ib,i)=den_arr(n)/pi
                    kk = 200.d0*mu*e(n)/hbc**2;  fact=4.*pi*abs(ka)/kk ;    cs(ib,i) = fact*sin(Sp_t)**2
                end if
             end do
             deallocate(den_arr)
            exit
        end do !nc
    end do !kap
        open(44, file = 'DCS/'//name(1:IFN)//tex%tit(it)//'.CS.'//pset%txtfor, status = 'unknown')
    write(44, '(2a12,30a20)') 'Ecm','Incident.E', 'Total.cs.barn',(lev(it)%tb(chunk(it)%ia(ib)+1), ib = 1, chunk(it)%nb )

    do ib = 1, chunk(it)%nb
        cs(NTX+1,:)=cs(NTX+1,:)+cs(ib,:)
    end do
        do i=0,Nest
            n=i*j
            write(44,'(1f12.6,30Es20.8)')e(n),e(n)*EcmInc,cs(NTX+1,i),(cs(ib,i),ib=1,chunk(it)%nb)
        end do
    close(44)

    open(44, file = 'DCS/'//name(1:IFN)//tex%tit(it)//'.LD.'//pset%txtfor, status = 'unknown')
    write(44, '(a12,30a20)') 'Ecm',(lev(it)%tb(chunk(it)%ia(ib)+1), ib = 1, chunk(it)%nb )
        do i=0,Nest
            n=i*j
            write(44,'(1f12.6,30Es20.8)')e(n),(LD(ib,i),ib=1,chunk(it)%nb)
        end do
    close(44)

    open(44, file = 'DCS/'//name(1:IFN)//tex%tit(it)//'.SP.'//pset%txtfor, status = 'unknown')
    write(44, '(a12,30a20)') 'Ecm',(lev(it)%tb(chunk(it)%ia(ib)+1), ib = 1, chunk(it)%nb )
        do i=0,Nest
            n=i*j
            write(44,'(1f12.6,30Es20.8)')e(n),(SP(ib,n),ib=1,chunk(it)%nb)
        end do
    close(44)


    !differential cross section
    lmax=minval(KMX); p=0.d0; lp=0.d0
    call gh_legendre(lmax, p, lp)
    call check_ortho(lmax, p)
    open(44, file = 'DCS/'//name(1:IFN)//tex%tit(it)//'.dcs.'//pset%txtfor, status = 'unknown')
    write(44, '(2a12,a20,61i20)') 'Ecm','Incident.E', 'Total', (3*i, i=0,60)
    do n=0, Nestj, j
        k = sqrt(2.d2*mu*e(n))/hbc ! d2 achieves the transformation of units from fm to sqrt(barns).
        g=zero; h=zero!; p=zero; lp=zero
        do ib = 1, chunk(it)%nb
            ka          = chunk(it)%kbl(ib) 
            l           = chunk(it)%l(ib)
            !write(*,'(a, 4i3)')'Legendre',l,ka,ib, chunk(it)%nb
            ss = abs(ka) * (exp(2.0d0 * ui * sp(ib,n)) - 1.0d0) / (2.0d0 * ui * k) 
            do i = 0, 60
                g(i) = g(i) + ss * p(l, i)
            end do
        end do
        do l = 1, lmax
            ka = -l-1
            !write(*,'(a, 4i3)')'Legendre',l,ka,lmax
            ibu = chunk(it)%ib(ka); ibd = chunk(it)%ib(l)
            !write(*,'(a, 4i3)')'Legendre',l,ka,ibu,ibd
            ss  = (exp(2.d0*ui*sp(ibu,n))-exp(2.d0*ui*sp(ibd,n)))/(2.d0*ui*k)
            do i = 0, 60
                h(i)=h(i)+ ss * lp(l,i)
            end do
        end do
        dcs(0:60) = abs(g(0:60))**2 + abs(h(0:60))**2
        integral = 0.d0
        ! 梯形法则：步长 3° → dθ = 3*pi/180
        do i = 0, 60
           theta = i * pi * 3.d0 / 180.d0
           s = sin(theta)
           ! 端点权重 0.5，内部点权重 1
           if (i == 0 .or. i == 60) then
              integral = integral + 0.5d0 * dcs(i)  * s
           else
              integral = integral +       dcs(i)  * s
           end if
        end do
        integral = integral * (pi * 3.d0 / 180.d0) * 2. * pi  ! 乘以步长和 2π（因为 dσ/dΩ 是每单位立体角的微分截面） 

        write(44,'(2f12.6,133Es20.8)')e(n),e(n)*EcmInc, &  !cs(NTX+1,n/j),integral, &
        & 4.*pi/k*(aimag(g(0))),(dcs(i),i=0,60)
    end do 
    close(44)

    if (inEnum .gt. 0) then
        allocate(kin(inEnum), dcsE(inEnum,0:60),inE_idx(inEnum), hl(inEnum,0:60), gk(inEnum,0:60))
        dcsE=zero; inE_idx=0.d0; kin=0.d0
        do n=1,inEnum
            inE_idx(n) = inE(n)/EcmInc/step
            k = sqrt(2.d2*mu*e(inE_idx(n)))/hbc ! d2 achieves the transformation of units from fm to sqrt(barns).
            do i = 0, 60
                gk(n,i)=zero; hl(n,i)=zero
                do ib = 1, chunk(it)%nb
                    ka          = chunk(it)%kbl(ib) 
                    l           = chunk(it)%l(ib)
                    ss = abs(ka) * (exp(2.0d0 * ui * sp(ib,inE_idx(n))) - 1.0d0) / (2.0d0 * ui * k) 
                    gk(n,i) = gk(n,i) + ss * p(l, i)
                end do !kappa
                do l = 1, lmax
                    ka = -l-1
                    ibu = chunk(it)%ib(ka); ibd = chunk(it)%ib(l)
                    ss  = (exp(2.d0*ui*sp(ibu,inE_idx(n))) - exp(2.d0*ui*sp(ibd,inE_idx(n))))/(2.d0*ui*k)
                    hl(n,i)=hl(n,i)+ ss * lp(l,i)
                end do! l
                dcsE(n,i) = abs(gk(n,i))**2 + abs(hl(n,i))**2
                !write(*,*)'n,i,dcsE=',n,i,dcsE(n,i)
            end do !i
        integral = 0.d0
        ! 梯形法则：步长 3° → dθ = 3*pi/180
        do i = 0, 60
           theta = i * pi * 3.d0 / 180.d0
           s = sin(theta)
           ! 端点权重 0.5，内部点权重 1
           if (i == 0 .or. i == 60) then
              integral = integral + 0.5d0 * dcsE(n,i)  * s
           else
              integral = integral +       dcsE(n,i)  * s
           end if
        end do
        kin(n) = integral * (pi * 3.d0 / 180.d0) * 2. * pi  ! 乘以步长和 2π（因为 dσ/dΩ 是每单位立体角的微分截面）
        write(*,*)'incident E, CS=',e(inE_idx(n))*EcmInc,kin(n) 
        end do !n
        
        
        open(44, file = 'DCS/'//name(1:IFN)//tex%tit(it)//'.dcsE.'//pset%txtfor, status = 'unknown')
        write(44, '(a12,100f20.6)') 'theta_Ecm',(e(inE_idx(n)),n=1,inEnum)
        write(44, '(a12,100f20.3)') 'Incident.E', (e(inE_idx(n))*EcmInc,n=1,inEnum)
        write(44, '(a12,100f20.6)') 'CS', (kin(n),n=1,inEnum)
        do i = 0, 60
            write(44,'(i6, 100Es20.8)') 3*i, (dcsE(n,i),n=1,inEnum)
        end do
        close(44)
        deallocate(inE, kin, dcsE, inE_idx, hl, gk)
    end if
    end do !it
    
    deallocate(sp,e,cs,LD)
    return

end subroutine DCSEl
!                                                                                                 !
! !                                                                                                 !
! Subroutine Satter                                                                                !
! !                                                                                                 !
! !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
! !
!     integer :: ib, i0, n0, nmax, n, it, i, ip, Nest, Nest1
!     integer :: ka, nc, max_pos, nn, j, nj
!     double precision :: er, ei, ResEl,ResEr, DeltaE, step_res, step, fj, IncE, EcmInc
!     double precision :: C, estp, E_min, E_max, prec, denstas, LDr, Sp_t, Sp_f, Sp_r! Scattering phase shift Total, Free, Delta
!     double precision :: M, mu, fact, kk, cs_t, cs_f, cs_r
!     double precision, allocatable :: cs_tt(:),cs_ft(:),cs_rt(:)!Sp_t(:), Sp_f(:), Sp_r(:)
!     step=0.001
!     Nest=ecut/step
!     allocate(cs_tt(0:Nest),cs_ft(0:Nest),cs_rt(0:Nest))
!     M=nuc%npr(1)*pset%amu(1)+nuc%npr(2)*pset%amu(2)+nuc%etot
    
!     XF=zero ;   XG=zero;  YF=zero;     YG=zero
!     XFh=zero;   XGh=zero;   YFh=zero;   YGh=zero
! !--- Loop over neutron and proton
!     Call system("mkdir -p DCS")
!     do it = 1, 2
!     if (.not.lprNP(it)) cycle
!     cs_tt=0.d0;     cs_ft=0.d0;   cs_rt=0.d0
!     mu=M*pset%amu(it)/(M+pset%amu(it))
!     EcmInc=(M+pset%amu(it))/M
!     !write(*,*)'Incident Energy in CM:',EcmInc,mu,M,pset%amu(it),nuc%etot,nuc%npr(it)
!     !--- Loop over the single particle levels
!     do ib = 1, chunk(it)%nb
!         n0          = chunk(it)%ia(ib)
!         nmax        = chunk(it)%id(ib)
!         ka          = chunk(it)%kbl(ib) 
!         !if (ka.ne.5) cycle
!         fj          = 4.d-2*pi*abs(ka)! d-2 achieves the transformation of units from fm^2 to barns.
!         denstas=0.d0;  Sp_t=0.d0; Sp_f=0.d0; Sp_r=0.d0
!         do nc = 1, nmax
!             i0              = n0 + nc
!             if(lev(it)%ee(i0).lt.0.d0) then
!                 Sp_t=Sp_t+pi;  Sp_r=Sp_r+pi
!                 cycle
!             end if
!             prec = lev(it)%ee(i0) - aint(lev(it)%ee(i0)/step  + 1d-12) *step
!             ResEl=ecut+10; ResEr=0.d0
!             !if (lev(it)%hgam(i0).lt.25*step) then
!             if (lev(it)%lpG(i0)) then
!                 ResEl=lev(it)%ee(i0)-4*lev(it)%hgam(i0)
!                 ResEr=lev(it)%ee(i0)+4*lev(it)%hgam(i0)
!                 step_res     = lev(it)%hgam(i0)/25
!                 if (ResEl.lt.0.d0) ResEl=prec
!                 if (step_res.gt.step) step_res=step
!                 nj = 100
!             end if
!             ! end if
!             ! if (lev(it)%hgam(i0).lt.1.d-5) then
!             !     ResEl=lev(it)%ee(i0)-step
!             !     ResEr=lev(it)%ee(i0)+step
!             !     step_res     = lev(it)%hgam(i0)
!             !     nj = 2*step/lev(it)%hgam(i0)/20
!             ! end if

            
!             !--- Calculate XG, XF, YG, YF
!             if(IE.eq.2) call GFDetXY(i0,it,.false.)   !--- Calculate XG, XF, YG, YF
!             open(44, file = 'DCS/'//name(1:IFN)//tex%tit(it)//'.'//lev(it)%tb(i0)(4:6)//'.2.'//pset%txtfor, status = 'unknown')
!             !write(44, '(2a12,10a20)') 'Ecm','Incident.E','Spt','Spf','Spd','cs_t.barn','cs_f.barn','cs_r.barn',lev(it)%tb(i0)
!             write(44, '(2a12,10a20)') 'Ecm','Incident.E','dCLD','Spd','cs_t.barn',lev(it)%tb(i0)
            
!             Er = prec; n=0
!             kk = 2.d0*mu*Er/hbc**2
!             fact=fj/kk 
!             denstas = abs(CLD(it,ka,er,1.d-6)) - abs(CLD0(it,ka,er,1.d-6))
!             !LDr = lev(it)%hgam(i0)/((er-lev(it)%ee(i0))**2 + (lev(it)%hgam(i0)/2)**2)
!             Sp_t=Sp_t+denstas*step!; Sp_f=Sp_f+(denstas-LDr)*step; Sp_r=Sp_r+LDr*step
!             cs_t = fact*sin(Sp_t)**2!;  cs_f = fact*sin(Sp_f)**2
!             !write(44,'(2f12.6,9Es20.8)')er,er*(M+pset%amu(it))/M,Sp_r,Sp_f,Sp_t,cs_r,cs_f,cs_t
!             write(44,'(2f12.6,9Es20.8)')er,er*EcmInc,denstas/pi,Sp_t,cs_t
!             cs_tt(0)=cs_tt(0)+cs_t;     cs_ft(0)=cs_ft(0)+cs_f;   cs_rt(0)=cs_rt(0)+cs_r
!             do while(Er.lt.ecut)!n=1,Nest
!                 Er = Er + step 
!                 kk = 2.d0*mu*Er/hbc**2
!                 fact=fj/kk
!                 j =0; n=n+1
!                 !if (abs(Er-lev(it)%ee(i0)).lt.0.1)write(*,*)'Near Resonance:',lev(it)%tb(i0),Er,ResEl,ResEr
!                 do while(Er.gt.ResEl .and. Er.lt.ResEr) 
!                     j = j + 1
!                     Er = ResEl + j*step_res
!                     kk = 2.d0*mu*Er/hbc**2
!                     fact=fj/kk
!                     denstas = abs(CLD(it,ka,er,1.d-6)) - abs(CLD0(it,ka,er,1.d-6))
!                     ! if (denstas.gt.0.d0) then
!                     !     Sp_r=Sp_r+denstas*step_res
!                     ! else
!                     !     Sp_f=Sp_f+denstas*step_res
!                     ! end if
!                     !Sp_t=Sp_t+denstas*step_res
!                     !LDr = lev(it)%hgam(i0)/((er-lev(it)%ee(i0))**2 + (lev(it)%hgam(i0)/2)**2)
!                     Sp_t=Sp_t+denstas*step_res!; Sp_f=Sp_f+(denstas-LDr)*step_res; Sp_r=Sp_r+LDr*step_res
!                     cs_t = fact*sin(Sp_t)**2!;  cs_r = fact*sin(Sp_r)**2;  cs_f = fact*sin(Sp_f)**2
!                     if(MOD(j,nj).eq.0) write(44,'(2f12.6,9Es20.8)')er,er*EcmInc,denstas/pi,Sp_t,cs_t!write(44,'(2f12.6,9Es20.8)')er,er*(M+pset%amu(it))/M,Sp_r,Sp_f,Sp_t,cs_r,cs_f,cs_t
!                 end do
!                 denstas = abs(CLD(it,ka,er,1.d-6)) - abs(CLD0(it,ka,er,1.d-6))
!                 !LDr = lev(it)%hgam(i0)/((er-lev(it)%ee(i0))**2 + (lev(it)%hgam(i0)/2)**2)
!                 Sp_t=Sp_t+denstas*step!; Sp_f=Sp_f+(denstas-LDr)*step; Sp_r=Sp_r+LDr*step
!                 !Sp_t=Sp_t+denstas*step;   Sp_f=Sp_f+denstas*step
!                 cs_t = fact*sin(Sp_t)**2!;  cs_r = fact*sin(Sp_r)**2;  cs_f = fact*sin(Sp_f)**2
                
!                 if(MOD(n,100).eq.0) write(44,'(2f12.6,9Es20.8)')er,er*EcmInc,denstas/pi,Sp_t,cs_t!write(44,'(2f12.6,9Es20.8)')er,er*(M+pset%amu(it))/M,Sp_r,Sp_f,Sp_t,cs_r,cs_f,cs_t
!                 cs_tt(n)=cs_tt(n)+cs_t;     cs_ft(n)=cs_ft(n)+cs_f;   cs_rt(n)=cs_rt(n)+cs_r
!         !        if (j.eq.0)cycle     
              
!         ! 20  if (E_max.ge.lev(it)%ee(i0))  THEN
!         !         i0=i0+1!; nc=nc+1
!         !         if (nc.gt.chunk(it)%id(ib)) exit
!         !             if(IE.eq.2) call GFDetXY(i0,it,.false.)
!         !         end if
!         !         denstas=0.d0
!         !         prec = lev(it)%ee(i0) - aint(lev(it)%ee(i0) * 100.0d0 + 1d-12) / 100.0d0
!              end do
            
!             close(44)
!             exit
!         end do !nc
!     end do !kap
!     open(44, file = 'DCS/'//name(1:IFN)//tex%tit(it)//'.'//'.TOT.'//pset%txtfor, status = 'unknown')
!         write(44, '(1a12,10a20)') 'Er','cs_t.barn','cs_f.barn','cs_r.barn'
!         do n=0,Nest
!                 Er = n*step 
!                 write(44,'(f12.6,9Es20.8)')er,cs_tt(n),cs_ft(n),cs_rt(n)
!         end do
!     close(44)
! end do!it
! !deallocate(cs_tt,cs_ft,cs_rt)
!     return
! !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
! !                                                                                                 !
! End Subroutine Satter                                                                          !
! !                                                                                                 !

!*************************************************************************************************!
subroutine ImGF
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!     This subroutine interpolates the Green's function
!
!-------------------------------------------------------------------------------------------------!
    complex*16 :: Ec, ui
    integer :: i, j, kk, ib, it, ka, npt, m0, max
    !integer, dimension(3) :: values = (/ -1, 5 , -6/)
    double precision :: C
    double precision, dimension(2) :: Ecarray != (/0.2d0, 1.5d0, 6.d0, 20.d0/)
    double precision, dimension(MSD) :: fun, funVS0
    double precision, dimension(3,MSD) :: sta, staVS0
    !double precision, dimension(3,MSD) :: den, denVS0
    integer :: n, nc, nmax, i0
    integer :: unit, ios
    character(len=100) :: line  ! 声明足够长的字符变量来存储行内容


    npt=well%npt;   ui = (0.d0, 1.d0);   it=1 
        ! 检查文件是否存在

        ! 打开文件
        open(newunit=unit, file='ImGF.dat', status='old', action='read', iostat=ios)
        if (ios /= 0) then
            print *, '无法打开文件: ', 'ImGF.dat'
            stop
        end if

        ! 逐行读取文件
        do
            ! 先读取整行
            read(unit, '(A)', iostat=ios) line
            ! 处理文件结束或错误
            if (ios /= 0) exit
            ! 去除前导空白并检查是否为空行
            if (len_trim(adjustl(line)) == 0) exit
            ! 从行内容中解析数据
            read(line, '(2I4,2f8.4)', iostat=ios) ka, nc, Ecarray
            ! 处理解析错误
            if (ios /= 0) exit
            
        sta=0.d0;     staVS0=0.d0;  C = 1.d0
        ib  = chunk(it)%ib(ka)
        i0  = chunk(it)%ia(ib) + nc
        write(*, '(a10,2I4,2f8.3)') lev(it)%tb(i0), ka, nc, Ecarray
        open(20, file = name(1:IFN)//lev(it)%tb(i0)(1:6)//'.'//pset%txtfor, status = 'unknown')
        !write(20, '(A6)', advance='no') 'r'
            if (.not.lev(it)%lpG(i0))exit 
                if(IE.eq.2) call GFDetXY(i0,it,.false.)
                if (lev(it)%ee(i0).lt.0.d0) then
                    Ec = lev(it)%ee(i0)-ui*1.d-6
                    call DiracGF(it,ka,Ec,.false.)
                    DOS%HGG(1)  = 3.*(DOS%HGG(2) - DOS%HGG(3)) + DOS%HGG(4)  
                    DOS%HFF(1)  = 3.*(DOS%HFF(2) - DOS%HFF(3)) + DOS%HFF(4)
                    fun=aimag(DOS%HGG)+aimag(DOS%HFF)
                    call simps(fun, npt, well%h, C)
                    staVS0(1,:)=0.d0; sta(1,:)=fun/C

                    write(*,*)lev(it)%tb(i0), Ec, lev(it)%ee(i0)
                else
                    Ec = lev(it)%ee(i0)
                    call DiracGF(it,ka,Ec,.false.) 
                    fun=-aimag(DOS%HGG)-aimag(DOS%HFF)
                    call simps(fun, npt, well%h, C)
                    sta(1,:)=fun/C
                    call DiracGFVS0(it,ka,Ec,.false.)
                    fun=-aimag(DOS%HGG)-aimag(DOS%HFF)
                    call simps(fun, npt, well%h, C)
                    staVS0(1,:)=fun/C
                end if
                ! sta(n,1) = 3.*(sta(n,2)-sta(n,3))+ sta(n,4)
                ! staVS0(n,1) = 3.*(staVS0(n,2)-staVS0(n,3))+ staVS0(n,4)
                write(20, '(A14, 2f14.3)', advance='no') lev(it)%tb(i0), lev(it)%ee(i0), lev(it)%hgam(i0)
                
            do j = 1, size(Ecarray)
                Ec = Ecarray(j)
                call DiracGF(it,ka,Ec,.false.) 
                sta(j+1,:)=-aimag(DOS%HGG)-aimag(DOS%HFF)
                call DiracGFVS0(it,ka,Ec,.false.)
                staVS0(j+1,:)=-aimag(DOS%HGG)-aimag(DOS%HFF) 
                write(20, '(A14, f14.3)', advance='no') lev(it)%tb(i0), real(Ec)
            end do 
            write(20,*) ! 最后手动换行
            do i=2,npt
                write(20,'(f6.3,20f14.6)')well%xr(i), (sta(m0,i),staVS0(m0,i), m0 = 1, 3)
            end do  
            close(20)
        end do

        ! 关闭文件
        close(unit)
!-------------------------------------------------------------------------------------------------!
!                                                                                                 !
End Subroutine ImGF                                                                                !
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
Subroutine Two_dimensional_density_states

    integer:: it,ka,nc
    integer :: n, Nest, nestg, ni, max_pos, ng, ne(2), i0, ib, mi
    double precision :: er, ei, estpge, C,ermin,eimin,free,eig,hgam,estpne,lg
    double precision, allocatable :: denstas(:,:)
    integer :: unit, ios
    character(len=100) :: line  ! 声明足够长的字符变量来存储行内容

    open(newunit=unit, file='2D_DOS.dat', status='old', action='read', iostat=ios)
        if (ios /= 0) then
            print *, '无法打开文件: ', '2D_DOS.dat'
            stop
        end if

do
    ! 先读取整行
    read(unit, '(A)', iostat=ios) line
    ! 处理文件结束或错误
    if (ios /= 0) exit
    ! 去除前导空白并检查是否为空行
    if (len_trim(adjustl(line)) == 0) exit
    ! 从行内容中解析数据
    read(line, *, iostat=ios) it, ka, nc, mi, estpne, eig, hgam
    write(*, '(4I4, f8.6, 2f8.4)', iostat=ios) it, ka, nc, mi, estpne, eig, hgam
 
    ! 处理解析错误
    if (ios /= 0) exit
    
    ib  = chunk(it)%ib(ka);     i0  = chunk(it)%ia(ib) + nc
    Nest =200;   Nestg=200
    write(*,'(2a,2f12.8,Es20.8)')lev(it)%tb(i0),'er,ei,DOS=',eig,hgam,lev(it)%ee(i0) 
    if (lev(it)%lpG(i0))  THEN
        eig = my_round(lev(it)%ee(i0),mi)
        hgam= my_round(lev(it)%hgam(i0),mi)
    end if
    write(*,'(2a,2f12.8,Es20.8)')lev(it)%tb(i0),'er,ei,DOS=',eig,hgam

    if(IE.eq.2) call GFDetXY(i0,it,.false.)  
    
    ermin=eig - nest*estpne/2.d0
    eimin=hgam - nestg*estpne/200.d0
     if (ermin .lt. 0.d0) ermin=1.d-6
     !if (eimin .lt. 0.d0) eimin=1.d-12
     
     write(*, '(2a10,3I4, 3f12.8,a,2Es20.7)') name(1:IFN)//tex%tit(it)//'.'//lev(it)%tb(i0)(4:6), lev(it)%tb(i0), it, ka, nc, estpne, eig, hgam,'ermin', ermin, eimin
  !  ei=0.d0; ni=0
    open(44, file = name(1:IFN)//tex%tit(it)//'.'//lev(it)%tb(i0)(4:6)//'.two.'//pset%txtfor, status = 'unknown')
    write(44, '(2a12,4a20)') 'Er','Gam/2','N_total','free','total-free',lev(it)%tb(i0)

    allocate(denstas(0:Nest,0:Nestg))
    denstas=0.d0
        do n=0,Nest
            Er=ermin + n*estpne 
            do ng=0, nestg
                ei= eimin+ng*estpne/100
                denstas(n,ng)=CLD(it,ka,er,ei)
                free=CLD0(it,ka,er,ei)
                ! if (abs(denstas(n,ng)).lt.1.d0) then
                !     cycle 
                !     lg=denstas(n,ng)/10.d0
                ! else
                !     lg=log10(abs(denstas(n,ng)))
                !     if (denstas(n,ng).lt.0.d0) lg=-lg
                ! end if
                write(44,'(2f12.8,4Es20.8)')er,-ei,denstas(n,ng),free,denstas(n,ng)-free
            end do
        end do
    ne  = maxloc(abs(denstas))
    Er=ermin + ne(1)*estpne-estpne
    ei=eimin + ne(2)*estpne-estpne
    !call Check_density(it,i0,nc,ka,er,ei,denstas(ne(1),ne(2)),C)
    write(*,'(2a,2f12.8,Es20.8)')lev(it)%tb(i0),'er,ei,DOS=',er,ei,denstas(ne(1)-1,ne(2)-1)
    deallocate(denstas)
    close(44)
end do
End Subroutine Two_dimensional_density_states

subroutine Runge_Kutta(it, ka, eigfm, gin, fin, gout, fout)
    integer, intent(in) :: it, ka
    complex*16, intent(in) :: eigfm
    complex*16 , intent(inout) :: gin(msd),fin(msd),gout(msd),fout(msd)
    complex*16, dimension(4) :: ag, af
    complex*16  sg1, sf1, sg2, sf2, u1g, u1f, u2g, u2f, u1x, u2y
    double precision :: h, h2, r, r1, r2, r4
    integer :: i, jk, npt, j
    
    h      = well%h;               h2   = well%h*half;           npt = well%npt
    do i  = 2, npt-1
        r   = well%xr(i)
        r1  = ka/ r
        r2  = ka/(r + h2)
        r4  = ka/(r + h)
      
        sg1 = gin(i)
        sf1 = fin(i)

        do  jk = 1, 4
            go to (46,47,47,48) , jk
 46         sg2     = sg1;                                      sf2     = sf1
            u1g     = r1  + dpotl(it)%vt(i) + XG(i);            u1f     = eigfm - dpotl(it)%vms(i) - XF(i)
            u2f     = r1  + dpotl(it)%vt(i) + YF(i);            u2g     = eigfm - dpotl(it)%vps(i) - YG(i)
            go to 45

 47         sg2     = sg1 + ag(jk-1);                           sf2     = sf1 + af(jk-1)
            u1g     = r2  + dpotl(it)%vth(i) + XGh(i);          u1f     = eigfm - dpotl(it)%vmsh(i) - XFh(i)
            u2f     = r2  + dpotl(it)%vth(i) + YFh(i);          u2g     = eigfm - dpotl(it)%vpsh(i) - YGh(i)
            go to 45

 48         sg2     = sg1 + two*ag(3);                          sf2     = sf1 + two*af(3)
            u1g     = r4  + dpotl(it)%vt(i+1) + XG(i+1);        u1f     = eigfm - dpotl(it)%vms(i+1) - XF(i+1)
            u2f     = r4  + dpotl(it)%vt(i+1) + YF(i+1);        u2g     = eigfm - dpotl(it)%vps(i+1) - YG(i+1)

 45         ag(jk)  = h2*(-u1g*sg2 + u1f*sf2);                  af(jk)  = h2*( u2f*sf2 - u2g*sg2)
        end do
        sg2     = (ag(1) + two*(ag(2)+ag(3)) + ag(4))*third;    sf2     = (af(1) + two*(af(2)+af(3)) + af(4))*third
        gin(i+1) = sg1 + sg2;                                   fin(i+1) = sf1 + sf2
    end do ! end iteration outward to 0 point.
            
    do i = npt,3,-1
        r   = well%xr(i)
        r1  = ka/ r
        r2  = ka/(r - h2)  ! back half step
        r4  = ka/(r - h)       ! back one step

        sg1 = gout(i);            sf1 = fout(i)
        
        do  jk = 1, 4
            go to (36,37,37,38) , jk     ! go to(36) is for jk=1, go to(37) for jk=2 and 3, go to (38) for jk=4
 36         sg2     = sg1;                                      sf2     = sf1
            u1g     = r1  + dpotl(it)%vt(i) + XG(i);            u1f     = eigfm - dpotl(it)%vms(i) - XF(i)
            u2f     = r1  + dpotl(it)%vt(i) + YF(i);            u2g     = eigfm - dpotl(it)%vps(i) - YG(i)
            go to 35

 37         sg2     = sg1 + ag(jk-1);                           sf2     = sf1 + af(jk-1)
            u1g     = r2  + dpotl(it)%vth(i-1) + XGh(i-1);      u1f     = eigfm - dpotl(it)%vmsh(i-1) - XFh(i-1)
            u2f     = r2  + dpotl(it)%vth(i-1) + YFh(i-1);      u2g     = eigfm - dpotl(it)%vpsh(i-1) - YGh(i-1)
            go to 35

 38         sg2     = sg1 + two*ag(3);                          sf2     = sf1 + two*af(3)
            u1g     = r4  + dpotl(it)%vt(i-1) + XG(i-1);        u1f     = eigfm - dpotl(it)%vms(i-1) - XF(i-1)
            u2f     = r4  + dpotl(it)%vt(i-1) + YF(i-1);        u2g     = eigfm - dpotl(it)%vps(i-1) - YG(i-1)
         
 35         ag(jk)  = - h2*(-u1g*sg2 + u1f*sf2);                af(jk)  = - h2*( u2f*sf2 - u2g*sg2)
        end do
        sg2     = (ag(1) + two*(ag(2)+ag(3)) + ag(4))*third;    sf2     = (af(1) + two*(af(2)+af(3)) + af(4))*third
        gout(i-1) = sg1 + sg2;                                  fout(i-1) = sf1 + sf2
    end do
    ! gout(1)  = 3.*(gout(2)- gout(3)) + gout(4)
    ! fout(1)  = 3.*(fout(2)- fout(3)) + fout(4)
end subroutine Runge_Kutta

subroutine Runge_Kutta_Free(it, ka, eigfm, gin, fin, gout, fout)
    integer, intent(in) :: it, ka
    complex*16, intent(in) :: eigfm
    complex*16 , intent(inout) :: gin(msd),fin(msd),gout(msd),fout(msd)
    complex*16, dimension(4) :: ag, af
    complex*16  sg1, sf1, sg2, sf2, u1g, u1f, u2g, u2f, u1x, u2y
    double precision :: h, h2, r, r1, r2, r4
    integer :: i, jk, npt, j
    double precision, dimension(2) :: tc
    data tc/0.0d0, 1.0d0/

    h      = well%h;               h2   = well%h*half;           npt = well%npt
    ! Vcou   = 0.d0;             Vcouh   = 0.d0
    ! if (it.eq.2) THEN
    !     Vcou=dself%cou
    !     Call intpol6(Vcou, Vcouh, npt)
    ! end if
    !--- Integrate inward from (npt - 1) *mesh
    do i  = 2, npt-1
        r   = well%xr(i)
        r1  = ka/ r
        r2  = ka/(r + h2)
        r4  = ka/(r + h)
      
        sg1 = gin(i)
        sf1 = fin(i)

        do  jk = 1, 4
            go to (460,470,470,480) , jk
 460         sg2     = sg1;                                      sf2     = sf1
            u1g     = r1  !+ dpotl(it)%vt(i) + XG(i);            
            u1f     = eigfm - dpotl(it)%vms(i) !- XF(i)
            u2f     = r1  !+ dpotl(it)%vt(i) + YF(i);           
             u2g     = eigfm - tc(it)*dself%cou(i)!- dpotl(it)%vps(i) - YG(i)
            go to 450

 470         sg2     = sg1 + ag(jk-1);                           sf2     = sf1 + af(jk-1)
            u1g     = r2  !+ dpotl(it)%vth(i) + XGh(i);         
             u1f     = eigfm - dpotl(it)%vmsh(i) !- XFh(i)
            u2f     = r2  !+ dpotl(it)%vth(i) + YFh(i);          
            u2g     = eigfm - tc(it)*dself%couh(i)!- dpotl(it)%vpsh(i) - YGh(i)
            go to 450

 480         sg2     = sg1 + two*ag(3);                          sf2     = sf1 + two*af(3)
            u1g     = r4  !+ dpotl(it)%vt(i+1) + XG(i+1);        
            u1f     = eigfm - dpotl(it)%vms(i+1) !- XF(i+1)
            u2f     = r4  !+ dpotl(it)%vt(i+1) + YF(i+1);        
            u2g     = eigfm - tc(it)*dself%cou(i+1) !- dpotl(it)%vps(i+1) - YG(i+1)

 450         ag(jk)  = h2*(-u1g*sg2 + u1f*sf2);                  af(jk)  = h2*( u2f*sf2 - u2g*sg2)
        end do
        sg2     = (ag(1) + two*(ag(2)+ag(3)) + ag(4))*third;    sf2     = (af(1) + two*(af(2)+af(3)) + af(4))*third
        gin(i+1) = sg1 + sg2;                                   fin(i+1) = sf1 + sf2
    end do ! end iteration outward to 0 point.



                
    do i = npt,3,-1
        r   = well%xr(i)
        r1  = ka/ r
        r2  = ka/(r - h2)  ! back half step
        r4  = ka/(r - h)       ! back one step

        sg1 = gout(i);            sf1 = fout(i)
        
        do  jk = 1, 4
            go to (360,370,370,380) , jk     ! go to(36) is for jk=1, go to(37) for jk=2 and 3, go to (38) for jk=4
 360        sg2     = sg1;                                      sf2     = sf1
            u1g     = r1  !+ dpotl(it)%vt(i) + XG(i);            
            u1f     = eigfm - dpotl(it)%vms(i) !- XF(i)
            u2f     = r1  !+ dpotl(it)%vt(i) + YF(i);            
            u2g     = eigfm - tc(it)*dself%cou(i) !- dpotl(it)%vps(i) - YG(i)
            go to 350

 370        sg2     = sg1 + ag(jk-1);                           sf2     = sf1 + af(jk-1)
            u1g     = r2  !+ dpotl(it)%vth(i-1) + XGh(i-1);      
            u1f     = eigfm - dpotl(it)%vmsh(i-1) !- XFh(i-1)
            u2f     = r2  !+ dpotl(it)%vth(i-1) + YFh(i-1);     
            u2g     = eigfm - tc(it)*dself%couh(i-1)!- dpotl(it)%vpsh(i-1) - YGh(i-1)
            go to 350

 380        sg2     = sg1 + two*ag(3);                          sf2     = sf1 + two*af(3)
            u1g     = r4  !+ dpotl(it)%vt(i-1) + XG(i-1);        
            u1f     = eigfm - dpotl(it)%vms(i-1) !- XF(i-1)
            u2f     = r4  !+ dpotl(it)%vt(i-1) + YF(i-1);        
            u2g     = eigfm - tc(it)*dself%cou(i-1)!- dpotl(it)%vps(i-1) - YG(i-1)
         
 350         ag(jk)  = - h2*(-u1g*sg2 + u1f*sf2);                af(jk)  = - h2*( u2f*sf2 - u2g*sg2)
        end do
        sg2     = (ag(1) + two*(ag(2)+ag(3)) + ag(4))*third;    sf2     = (af(1) + two*(af(2)+af(3)) + af(4))*third
        gout(i-1) = sg1 + sg2;                                  fout(i-1) = sf1 + sf2
    end do
    ! gout(1)  = 3.*(gout(2)- gout(3)) + gout(4)
    ! fout(1)  = 3.*(fout(2)- fout(3)) + fout(4)
end subroutine Runge_Kutta_Free
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
subroutine spherical_hankel1(n, z, result)
    integer, intent(in) :: n
    complex*16, intent(in) :: z
    integer :: i
    complex*16 :: result, hankel_array(0:15)
    complex*16 :: ui
    ui=(0.d0, 1.d0)

! 计算复数自变量的第一类球Hankel函数的子程序

    if (n == 0) then
        result = -ui/z*exp(ui*z)
        return
    else if (n == 1) then
        result = -(ui/z**2+1./z)*exp(ui*z)
        return
    else
        hankel_array(0) = -ui/z*exp(ui*z)
        hankel_array(1) = -(ui/z**2+1./z)*exp(ui*z)
        do i = 2, n
            hankel_array(i) = cmplx(2.0*i - 1.0, 0.0) / z * hankel_array(i - 1) - hankel_array(i - 2)
        end do
        result = hankel_array(n)
    end if
end subroutine spherical_hankel1
!*************************************************************************************************!
! subroutine spherical_hankel1(n, z, result)
!     implicit none
!     integer, intent(in) :: n
!     complex*16, intent(in) :: z
!     complex*16, intent(out) :: result
!     integer :: kode, m, nz, ierr
!     complex*16 :: cy(1)
!     real*8 :: fnu
!     complex*16 :: ui
    
!     ui = (0.d0, 1.d0)
    
!     ! 第一类球Hankel函数与标准Hankel函数的关系:
!     ! h_n^(1)(z) = sqrt(pi/(2z)) * H_(n+1/2)^(1)(z)
    
!     if (abs(z) == 0.d0) then
!         ! 处理z=0的特殊情况
!         if (n == 0) then
!             result = (1.d0, 0.d0)  ! h_0^(1)(0) = 1
!         else
!             result = (0.d0, 0.d0)  ! 其他阶数为0
!         endif
!         return
!     endif
    
!     ! 设置CBESH参数
!     fnu = dble(n) + 0.5d0  ! 球Hankel函数对应半整数阶
!     kode = 1               ! 1: 未缩放的函数, 2: 指数缩放的函数
!     m = 1                  ! 1: 计算第一类Hankel函数H^(1)
!     nz = 0                 ! 输出参数: 未计算元素数
!     ierr = 0               ! 错误代码
    
!     ! 调用CBESH计算Hankel函数
!     call cbesh(z, fnu, kode, m, 1, cy, nz, ierr)
    
!     if (ierr /= 0) then
!         ! 处理错误情况
!         print *, 'CBESH error:', ierr
!         result = (0.d0, 0.d0)
!         return
!     endif
    
!     ! 转换为球Hankel函数: h_n^(1)(z) = sqrt(pi/(2z)) * H_(n+1/2)^(1)(z)
!     result = sqrt(pi/(2.d0*z)) * cy(1)
    
! end subroutine spherical_hankel1

! SUBROUTINE CBESH (Z, FNU, KODE, M, N, CY, NZ, IERR)
!   complex*16 :: CY, Z, ZN, ZT, CSGN
!   double precision ::  AA, ALIM, ALN, ARG, AZ, CPN, DIG, ELIM, FMM, FN, FNU, FNUL
!   double precision ::  HPI, RHPI, RL, R1M5, SGN, SPN, TOL, UFL, XN, XX, YN, YY, R1MACH
!   double precision ::  BB, ASCLE, RTOL, ATOL
!   INTEGER I, IERR, INU, INUH, IR, K, KODE, K1, K2, M, MM, MR, N, NN, NUF, NW, NZ, I1MACH
!   DIMENSION CY(N)
!   DATA HPI /1.57079632679489662E0/
!       NZ=0
!       XX = REAL(Z)
!       YY = AIMAG(Z)
!       IERR = 0
!       IF (XX.EQ.0.0E0 .AND. YY.EQ.0.0E0) IERR=1
!       IF (FNU.LT.0.0E0) IERR=1
!       IF (M.LT.1 .OR. M.GT.2) IERR=1
!       IF (KODE.LT.1 .OR. KODE.GT.2) IERR=1
!       IF (N.LT.1) IERR=1
!       IF (IERR.NE.0) RETURN
!       NN = N

!       TOL = MAX(R1MACH(4),1.0E-18)
!       K1 = I1MACH(12)
!       K2 = I1MACH(13)
!       R1M5 = R1MACH(5)
!       K = MIN(ABS(K1),ABS(K2))
!       ELIM = 2.303E0*(K*R1M5-3.0E0)
!       K1 = I1MACH(11) - 1
!       AA = R1M5*K1
!       DIG = MIN(AA,18.0E0)
!       AA = AA*2.303E0
!       ALIM = ELIM + MAX(-AA,-41.45E0)
!       FNUL = 10.0E0 + 6.0E0*(DIG-3.0E0)
!       RL = 1.2E0*DIG + 3.0E0
!       FN = FNU + (NN-1)
!       MM = 3 - M - M
!       FMM = MM
!       ZN = Z*CMPLX(0.0E0,-FMM)
!       XN = REAL(ZN)
!       YN = AIMAG(ZN)
!       AZ = ABS(Z)
!       AA = 0.5E0/TOL
!       BB=I1MACH(9)*0.5E0
!       AA=MIN(AA,BB)
!       IF(AZ.GT.AA) GO TO 240
!       IF(FN.GT.AA) GO TO 240
!       AA=SQRT(AA)
!       IF(AZ.GT.AA) IERR=3
!       IF(FN.GT.AA) IERR=3
!       UFL = R1MACH(1)*1.0E+3
!       IF (AZ.LT.UFL) GO TO 220
!       IF (FNU.GT.FNUL) GO TO 90
!       IF (FN.LE.1.0E0) GO TO 70
!       IF (FN.GT.2.0E0) GO TO 60
!       IF (AZ.GT.TOL) GO TO 70
!       ARG = 0.5E0*AZ
!       ALN = -FN*LOG(ARG)
!       IF (ALN.GT.ELIM) GO TO 220
!       GO TO 70
!    60 CONTINUE
!       CALL CUOIK(ZN, FNU, KODE, 2, NN, CY, NUF, TOL, ELIM, ALIM)
!       IF (NUF.LT.0) GO TO 220
!       NZ = NZ + NUF
!       NN = NN - NUF
!       IF (NN.EQ.0) GO TO 130
!    70 CONTINUE
!       IF ((XN.LT.0.0E0).OR.(XN.EQ.0.0E0 .AND. YN.LT.0.0E0 .AND. M.EQ.2)) GO TO 80
!       CALL CBKNU(ZN, FNU, KODE, NN, CY, NZ, TOL, ELIM, ALIM)
!       GO TO 110
!    80 CONTINUE
!       MR = -MM
!       CALL CACON(ZN, FNU, KODE, MR, NN, CY, NW, RL, FNUL, TOL, ELIM, ALIM)
!       IF (NW.LT.0) GO TO 230
!       NZ=NW
!       GO TO 110
!    90 CONTINUE
!       MR = 0
!       IF ((XN.GE.0.0E0).AND.(XN.NE.0.0E0 .OR. YN.GE.0.0E0 .OR. M.NE.2)) GO TO 100
!       MR = -MM
!       IF (XN.EQ.0.0E0 .AND. YN.LT.0.0E0) ZN = -ZN
!   100 CONTINUE
!       CALL CBUNK(ZN, FNU, KODE, MR, NN, CY, NW, TOL, ELIM, ALIM)
!       IF (NW.LT.0) GO TO 230
!       NZ = NZ + NW
!   110 CONTINUE

!       SGN = SIGN(HPI,-FMM)
!       INU = FNU
!       INUH = INU/2
!       IR = INU - 2*INUH
!       ARG = (FNU-(INU-IR))*SGN
!       RHPI = 1.0E0/SGN
!       CPN = RHPI*COS(ARG)
!       SPN = RHPI*SIN(ARG)
! !     ZN = CMPLX(-SPN,CPN)
!       CSGN = CMPLX(-SPN,CPN)
! !     IF (MOD(INUH,2).EQ.1) ZN = -ZN
!       IF (MOD(INUH,2).EQ.1) CSGN = -CSGN
!       ZT = CMPLX(0.0E0,-FMM)
!       RTOL = 1.0E0/TOL
!       ASCLE = UFL*RTOL
!       DO 120 I=1,NN
! !       CY(I) = CY(I)*ZN
! !       ZN = ZN*ZT
!         ZN=CY(I)
!         AA=REAL(ZN)
!         BB=AIMAG(ZN)
!         ATOL=1.0E0
!         IF (MAX(ABS(AA),ABS(BB)).GT.ASCLE) GO TO 125
!           ZN = ZN*CMPLX(RTOL,0.0E0)
!           ATOL = TOL
!   125   CONTINUE
!         ZN = ZN*CSGN
!         CY(I) = ZN*CMPLX(ATOL,0.0E0)
!         CSGN = CSGN*ZT
!   120 CONTINUE
!       RETURN
!   130 CONTINUE
!       IF (XN.LT.0.0E0) GO TO 220
!       RETURN
!   220 CONTINUE
!       IERR=2
!       NZ=0
!       RETURN
!   230 CONTINUE
!       IF(NW.EQ.(-1)) GO TO 220
!       NZ=0
!       IERR=5
!       RETURN
!   240 CONTINUE
!       NZ=0
!       IERR=4
!       RETURN
!  End subroutine CBESH

End module Greenf                                                                            !
!                                                                                                 !
