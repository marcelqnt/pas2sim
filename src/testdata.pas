
const
  EXPAPP_K_INC = 40;
  EXPAPP_K_DEC = 11;
  FLASH_TIME = 1.4;
  FLASH_ON_TIME = 0.7;

{PUBLIC_VARS
  $Invert: boolean;
  trafficlight_phase: integer;   
  signalroute_active: boolean;
  signalstate: integer;  
  light: single;  
}

function ExpApp(y0, y1: single): single;
begin
  if y1 > y0 then
    result := y0 + (y1-y0)*exp(-EXPAPP_K_DEC*Timegap)
  else
    result := y0 + (y1-y0)*exp(-EXPAPP_K_INC*Timegap);
end;

function ExpAppBool(y0: boolean; y1: single): single;
begin
  if y0 then
    result := ExpApp(1, y1)
  else
    result := ExpApp(0, y1);
end;

procedure SimStep;                
begin
  light := ExpAppBool((trafficlight_phase <= 5) and (not signalroute_active) and ((signalstate = 0) xor $Invert), light);
end;

procedure SimStep_LOD;
begin
  if (trafficlight_phase <= 5) and (not signalroute_active) and ((signalstate = 0) xor $Invert) then
    light := 1
  else
    light := 0;
end;

end.   
