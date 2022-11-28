import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import newton
from scipy.signal import argrelextrema
from collections import Iterable

palette = iter([
'#e43b37'
,'#fd931d'
,'#f7bc24'
,'#7fff00'
,'#5ebd3e'
,'#00ff7f'
,'#00ffff'
,'#16b0dd'
,'#007fff'
,'#974994'])

# Critical pressure, volume and temperature
# These values are for the van der Waals equation of state for CO2:
# (p - a/V^2)(V-b) = RT. Units: p is in Pa, Vc in m3/mol and T in K.
pc = 7.404e6
Vc = 1.28e-4
Tc = 304


def flatten(lis):
    """Flatten an irregular (arbitrarily nested) list of lists"""

    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


def vdw(Tr, Vr):
    """Van der Waals equation of state.

    Return the reduced pressure from the reduced temperature and volume.

    """

    pr = 8 * Tr / (3 * Vr - 1) - 3 / Vr ** 2
    return pr


def opt_vdw(Tr):
    """

    Return reduced pressure for different temperature where dP/dV is zero.

    """

    Vr, pr = symbols('Vr pr')
    X_opt = []

    pr = 8 * Tr / (3 * Vr - 1) - 3 / Vr ** 2
    Diff = diff(pr, Vr)
    result = solve(Eq(Diff, 0))
    for i in result:
        X_opt.append(re(i))

    X_opt.pop(0)

    return X_opt


def vdw_maxwell(Tr, Vr):
    """Van der Waals equation of state with Maxwell construction.

    Return the reduced pressure from the reduced temperature and volume,
    applying the Maxwell construction correction to the unphysical region
    if necessary.

    """

    pr = vdw(Tr, Vr)
    if Tr >= 1:
        # No unphysical region above the critical temperature.
        return [pr]

    if min(pr) < 0:
        raise ValueError('Negative pressure results from van der Waals'
                         ' equation of state with Tr = {} K.'.format(Tr))

    # Initial guess for the position of the Maxwell construction line:
    # the volume corresponding to the mean pressure between the minimum and
    # maximum in reduced pressure, pr.
    iprmin = argrelextrema(pr, np.less)
    iprmax = argrelextrema(pr, np.greater)
    Vr0 = np.mean([Vr[iprmin], Vr[iprmax]])

    def get_Vlims(pr0):
        """Solve the inverted van der Waals equation for reduced volume.

        Return the lowest and highest reduced volumes such that the reduced
        pressure is pr0. It only makes sense to call this function for
        T<Tc, ie below the critical temperature where there are three roots.

        """

        eos = np.poly1d((3 * pr0, -(pr0 + 8 * Tr), 9, -3))
        roots = eos.r
        roots.sort()
        Vrmin, _, Vrmax = roots
        return Vrmin, Vrmax

    def get_area_difference(Vr0):
        """Return the difference in areas of the van der Waals loops.

        Return the difference between the areas of the loops from Vr0 to Vrmax
        and from Vrmin to Vr0 where the reduced pressure from the van der Waals
        equation is the same at Vrmin, Vr0 and Vrmax. This difference is zero
        when the straight line joining Vrmin and Vrmax at pr0 is the Maxwell
        construction.

        """

        pr0 = vdw(Tr, Vr0)
        Vrmin, Vrmax = get_Vlims(pr0)
        return quad(lambda vr: vdw(Tr, vr) - pr0, Vrmin, Vrmax)[0]

    # Root finding by Newton's method determines Vr0 corresponding to
    # equal loop areas for the Maxwell construction.
    Vr0 = newton(get_area_difference, Vr0)
    pr0 = vdw(Tr, Vr0)
    Vrmin, Vrmax = get_Vlims(pr0)

    # Set the pressure in the Maxwell construction region to constant pr0.
    pr[(Vr >= Vrmin) & (Vr <= Vrmax)] = pr0
    return [pr, [Vrmin, Vrmax, Vr0, pr0]]


Vr = np.linspace(0.5, 3, 500)


def plot_pV(T1, T2):
    VrList = []
    Pr_List = []
    Pr_prime_List = []
    Optimum_pressure = []

    VrList.append(1)


    #The size of each step is 5 since we want to draw for 10 different tempreture b/w 270 and 320.
    Temperature = [temperature / Tc for temperature in range(T1, T2, 5) if temperature/Tc < 1]
    Temperature_prime = [temperature/ Tc for temperature in range(T1, T2, 5) if temperature/Tc < 1]


    for T in range(T1, T2, 5):

        Tr = T / Tc
        c = next(palette)
        if Tr <= 1:
            Optimum_pressure.append(opt_vdw(Tr))

        Optimum_pressure.append(opt_vdw(1))

        ax.plot(Vr, vdw(Tr, Vr), lw=2, alpha=0.3, color=c)
        ax.plot(Vr, vdw_maxwell(Tr, Vr)[0], lw=2, color=c, label='{:.3f}'.format(Tr))
        ax.plot(vdw_maxwell(Tr, Vr)[0][0], vdw_maxwell(Tr, Vr)[0][3], lw=2)

        for i in vdw_maxwell(Tr, Vr):

            if len(i) < 10:
                ax.fill_between(Vr, i[3], vdw(Tr, Vr), where=(Vr >= i[0]) & (Vr <= i[2]), facecolor=c, alpha=0.3)
                ax.fill_between(Vr, i[3], vdw(Tr, Vr), where=(Vr >= i[2]) & (Vr <= i[1]), facecolor=c, alpha=0.3)


                VrList.extend([i[0], i[1]])

                Pr_List.append(i[3])
                Pr_prime_List.append(i[3])


    VrList.sort()
    Pr_List.append(1)
    for i in Pr_prime_List:
        Pr_List.append(i)
    Pr_List[(len(Pr_prime_List) + 1):len(Pr_List)] = Pr_List[(len(Pr_prime_List) + 1):len(Pr_List)][::-1]

    # ORP is the list of the values of reduced volume for different tempreture where dP/dV = 0.
    ORV = list(flatten(Optimum_pressure))

    Vr_optimum = np.unique(ORV).tolist()

    Temperature.append(1)

    for i in Temperature_prime:
        Temperature.append(i)
    Temperature[(len(Temperature_prime) + 1):len(Temperature)] = Temperature[(len(Temperature_prime) + 1):len(Temperature)][::-1]

    Pr_optimum = [vdw(jj, j) for j, jj in zip(Vr_optimum, Temperature)]




    ax.plot(VrList, Pr_List, lw=2)
    ax.plot(Vr_optimum, Pr_optimum, lw=2)
    Binodal = [[redVol, redPres] for redVol, redPres in zip(VrList, Pr_List)]
    Spinodal = [[redVol, redPres] for redVol, redPres in zip(Vr_optimum, Pr_optimum)]


    print('Binodal = ', Binodal)

    print('Spinodal = ', Spinodal)


fig, ax = plt.subplots()

plot_pV(270, 320)

ax.set_xlim(0.4, 3)
ax.set_xlabel('Reduced volume')
ax.set_ylim(0, 1.6)
ax.set_ylabel('Reduced pressure')
ax.legend(title='Reduced temperature')

plt.show()
