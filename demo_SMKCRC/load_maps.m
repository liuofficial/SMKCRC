function maps = load_maps(n)
switch n,
    case 1, load priv_data\indian_map.mat;
    case 2, load priv_data\paviaU_map.mat;
    case 3, load priv_data\ksc_map.mat;
    case 4, load priv_data\paviaC_map.mat;
    case 5, load priv_data\paviaC_map.mat;
end
maps = map;
end