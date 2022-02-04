if pround(obj,digits) >= pround(obj0,digits) 
 % fprintf(1,'Warning: Finished line search without decreasing objective.\nMay have reached limit of precision, or obj/grad code may be broken.\n');
end
if etazero == eta 
 % fprintf(1,'Warning: Line search yielded no change in directional derivative.\nMay have reached limit of precision.\n');
end
if sum(x0+alpha.*direction ~= x0)==0 
 % fprintf(1,'Warning: Line search yielded no change in position.\nMay have reached limit of precision.\n');
end

% Changelog
% 8/22/06 - Changed print statement to suggest that obj/grad code may be broken
