from arpes.endstations import HemisphericalEndstation, FITSEndstation

__all__ = ('ALGMainChamber',)

class ALGMainChamber(HemisphericalEndstation, FITSEndstation):
    PRINCIPAL_NAME = 'ALG-Main'
    ALIASES = ['ALG-Main', 'ALG-MC', 'ALG-Hemisphere', 'ALG-Main Chamber',]
