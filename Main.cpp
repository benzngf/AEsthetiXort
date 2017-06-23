/*******************************************************************/
/*                                                                 */
/*                      ADOBE CONFIDENTIAL                         */
/*                   _ _ _ _ _ _ _ _ _ _ _ _ _                     */
/*                                                                 */
/* Copyright 2007 Adobe Systems Incorporated                       */
/* All Rights Reserved.                                            */
/*                                                                 */
/* NOTICE:  All information contained herein is, and remains the   */
/* property of Adobe Systems Incorporated and its suppliers, if    */
/* any.  The intellectual and technical concepts contained         */
/* herein are proprietary to Adobe Systems Incorporated and its    */
/* suppliers and may be covered by U.S. and Foreign Patents,       */
/* patents in process, and are protected by trade secret or        */
/* copyright law.  Dissemination of this information or            */
/* reproduction of this material is strictly forbidden unless      */
/* prior written permission is obtained from Adobe Systems         */
/* Incorporated.                                                   */
/*                                                                 */
/*******************************************************************/

/*	Main.cpp	

*/

#include "PixelSort_AE.h"
#define	FLOAT2FIX2(F)			((PF_Fixed)((F) * 65536 + (((F) < 0) ? -1 : 1)))

// CUDA
#include <cuda.h>

// forward declaration
extern "C"
int callCudaFunc();

bool hasCUDA;

static PF_Err 
About (	
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	suites.ANSICallbacksSuite1()->sprintf(	out_data->return_msg,
											"Hi :3 \n\nFor any questions, bug report\nor if you have some ideas about new features,\nplease contact me via e-mail\n  benzngf@gmail.com  \nThanks~~");

	return PF_Err_NONE;
}

static AEGP_PluginID myAEGPID;
static PF_Err 
GlobalSetup (	
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	out_data->my_version = PF_VERSION(	MAJOR_VERSION, 
										MINOR_VERSION,
										BUG_VERSION, 
										STAGE_VERSION, 
										BUILD_VERSION);
	out_data->out_flags = PF_OutFlag_NONE;
	out_data->out_flags2 = PF_OutFlag2_NONE;
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	suites.UtilitySuite5()->AEGP_RegisterWithAEGP(NULL, out_data->name,&myAEGPID);
	
	
return		PF_Err_NONE;

}

static PF_Err 
ParamsSetup (	
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_Err		err		= PF_Err_NONE;
	PF_ParamDef	def;	

	AEFX_CLR_STRUCT(def);
	out_data->num_params = 1;

	def.param_type = PF_Param_GROUP_START;
	PF_STRCPY(def.name, "Sorting Threshold...");
	def.uu.id = 1;
	def.flags = PF_ParamFlag_NONE;
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	A_char PopupStr_1[100];
	PF_STRCPY(PopupStr_1, "Hue|Saturation|Luminance|Red|Green|Blue");
	def.param_type = PF_Param_POPUP;
	PF_STRCPY(def.name, "Sort By");
	def.uu.id = 2;
	def.u.pd.num_choices = 6;
	def.u.pd.dephault = PSB_Luminance;
	def.u.pd.value = def.u.pd.dephault;
	def.u.pd.u.namesptr = PopupStr_1;
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_FLOAT_SLIDER;
	PF_STRCPY(def.name, "Min. Threshold");
	def.uu.id = 3;
	def.flags = 0;
	def.u.fs_d.precision = 1;
	def.u.fs_d.dephault = 100;
	def.u.fs_d.slider_min = 0;
	def.u.fs_d.slider_max = 100;
	def.u.fs_d.valid_max = 100;
	def.u.fs_d.valid_min = 0;
	def.u.fs_d.display_flags = PF_ValueDisplayFlag_PERCENT;
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_FLOAT_SLIDER;
	PF_STRCPY(def.name, "Max. Threshold");
	def.uu.id = 4;
	def.flags = 0;
	def.u.fs_d.precision = 1;
	def.u.fs_d.dephault = 100;
	def.u.fs_d.slider_min = 0;
	def.u.fs_d.slider_max = 100;
	def.u.fs_d.valid_max = 100;
	def.u.fs_d.valid_min = 0;
	def.u.fs_d.display_flags = PF_ValueDisplayFlag_PERCENT;
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_GROUP_END;
	def.uu.id = 5;
	def.flags = PF_ParamFlag_NONE;
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_CHECKBOX;
	PF_STRCPY(def.name, "");
	def.uu.id = 6;
	def.u.bd.dephault = true;
	A_char name[10];
	PF_STRCPY(name, "Reverse Sort Order");
	def.u.bd.u.nameptr = name;
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_GROUP_START;
	PF_STRCPY(def.name, "Sorting Pattern...");
	def.uu.id = 7;
	def.flags = PF_ParamFlag_NONE;
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	A_char PopupStr_2[100];
	PF_STRCPY(PopupStr_2, "Linear|Radial Spin|Polygon|Spiral|Sine|Triangle|Saw Tooth|Optical Flow");
	def.param_type = PF_Param_POPUP;
	PF_STRCPY(def.name, "Sorting Pattern");
	def.uu.id = 8;
	def.u.pd.num_choices = 8;
	def.u.pd.dephault = PSP_Linear;
	def.u.pd.value = def.u.pd.dephault;
	def.u.pd.u.namesptr = PopupStr_2;
	def.flags = PF_ParamFlag_SUPERVISE;  //use to change option parameters dynamically
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_GROUP_START;
	PF_STRCPY(def.name, "Pattern Options...");
	def.uu.id = 9;
	def.flags = PF_ParamFlag_NONE;
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_LAYER;
	PF_STRCPY(def.name, "Motion Reference Layer");
	def.uu.id = 10;
	def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
	//def.ui_flags = PF_PUI_NONE;
	def.ui_flags = PF_PUI_DISABLED; //hide this parameter(inactive)
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_ANGLE;
	PF_STRCPY(def.name, "Angle");
	def.uu.id = 11;
	def.u.ad.dephault = 0;
	def.u.ad.valid_max = 360000 << 16;
	def.u.ad.valid_min = -360000 << 16;
	def.ui_flags = PF_PUI_NONE;
	//def.ui_flags = PF_PUI_DISABLED; //hide this parameter(inactive)
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_FLOAT_SLIDER;
	PF_STRCPY(def.name, "Num. of Sides");
	def.uu.id = 18;
	def.flags = 0;
	def.u.fs_d.precision = 0;
	def.u.fs_d.dephault = 5;
	def.u.fs_d.slider_min =3;
	def.u.fs_d.slider_max = 50;
	def.u.fs_d.valid_max = 3;
	def.u.fs_d.valid_min = 50;
	def.u.fs_d.display_flags = PF_ValueDisplayFlag_NONE;
	//def.ui_flags = PF_PUI_NONE;
	def.ui_flags = PF_PUI_DISABLED; //hide this parameter(inactive)
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_POINT;
	PF_STRCPY(def.name, "Center");
	def.uu.id = 12;
	def.u.td.restrict_bounds = false;
	def.u.td.x_dephault =((A_long)in_data->width/2) << 16;
	def.u.td.y_dephault = ((A_long)in_data->height / 2) << 16;
	//def.ui_flags = PF_PUI_NONE;
	def.ui_flags = PF_PUI_DISABLED; //hide this parameter(inactive)
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_ANGLE;
	PF_STRCPY(def.name, "Curve Angle");
	def.uu.id = 13;
	def.u.ad.dephault = 0;
	def.u.ad.valid_max = 360000 << 16;
	def.u.ad.valid_min = -360000 << 16;
	//def.ui_flags = PF_PUI_NONE;
	def.ui_flags = PF_PUI_DISABLED; //hide this parameter(inactive)
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_FLOAT_SLIDER;
	PF_STRCPY(def.name, "Width/Height Ratio");
	def.uu.id = 14;
	def.flags = 0;
	def.u.fs_d.precision = 1;
	def.u.fs_d.dephault = 100;
	def.u.fs_d.slider_min = (PF_FpShort)0.1;
	def.u.fs_d.slider_max = 100;
	def.u.fs_d.valid_max = (PF_FpShort)100000;
	def.u.fs_d.valid_min = (PF_FpShort)0.1;
	def.u.fs_d.display_flags = PF_ValueDisplayFlag_PERCENT;
	//def.ui_flags = PF_PUI_NONE;
	def.ui_flags = PF_PUI_DISABLED; //hide this parameter(inactive)
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_FLOAT_SLIDER;
	PF_STRCPY(def.name, "Wave Length");
	def.uu.id = 15;
	def.flags = 0;
	def.u.fs_d.precision = 1;
	def.u.fs_d.dephault = 100;
	def.u.fs_d.slider_min = (PF_FpShort)2;
	def.u.fs_d.slider_max = 100;
	def.u.fs_d.valid_max = (PF_FpShort)100000;
	def.u.fs_d.valid_min = 2;
	def.u.fs_d.display_flags = PF_ValueDisplayFlag_NONE;
	//def.ui_flags = PF_PUI_NONE;
	def.ui_flags = PF_PUI_DISABLED; //hide this parameter(inactive)
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_FLOAT_SLIDER;
	PF_STRCPY(def.name, "Wave Height");
	def.uu.id = 16;
	def.flags = 0;
	def.u.fs_d.precision = 1;
	def.u.fs_d.dephault = 10;
	def.u.fs_d.slider_min = (PF_FpShort)0;
	def.u.fs_d.slider_max = 100;
	def.u.fs_d.valid_max = (PF_FpShort)5000;
	def.u.fs_d.valid_min = 0;
	def.u.fs_d.display_flags = PF_ValueDisplayFlag_NONE;
	//def.ui_flags = PF_PUI_NONE;
	def.ui_flags = PF_PUI_DISABLED; //hide this parameter(inactive)
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_ANGLE;
	PF_STRCPY(def.name, "Rotation");
	def.uu.id = 17;
	def.u.ad.dephault = 0;
	def.u.ad.valid_max = 360000 << 16;
	def.u.ad.valid_min = -360000 << 16;
	//def.ui_flags = PF_PUI_NONE;
	def.ui_flags = PF_PUI_DISABLED; //hide this parameter(inactive)
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_GROUP_END;
	def.uu.id = 98;
	def.flags = PF_ParamFlag_NONE;
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_GROUP_END;
	def.uu.id = 99;
	def.flags = PF_ParamFlag_NONE;
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_CHECKBOX;
	PF_STRCPY(def.name, "");
	def.uu.id = 100;
	def.u.bd.dephault = true;
	A_char name4[20];
	PF_STRCPY(name4, "Anti Aliasing");
	def.u.bd.u.nameptr = name4;
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_CHECKBOX;
	PF_STRCPY(def.name, "");
	def.uu.id = 101;
	def.u.bd.dephault = false;
	A_char name2[20];
	PF_STRCPY(name2, "Sort Alpha");
	def.u.bd.u.nameptr = name2;
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	def.param_type = PF_Param_CHECKBOX;
	PF_STRCPY(def.name, "Use GPU (Need CUDA Support)");
	def.uu.id = 102;
	def.u.bd.dephault = false;
	def.flags = PF_ParamFlag_CANNOT_TIME_VARY|PF_ParamFlag_SUPERVISE;
	A_char name3[30];
	PF_STRCPY(name3, "Use GPU");
	def.u.bd.u.nameptr = name3;
	if (err = PF_ADD_PARAM(in_data, -1, &def))
		return err;
	AEFX_CLR_STRUCT(def);
	out_data->num_params++;

	return err;
}



static PF_Err
ChangeParam(
PF_InData		*in_data,
PF_OutData  *out_data,
PF_ParamDef		*params[],
PF_LayerDef		*output,
PF_UserChangedParamExtra *extra){
	/*AEGP_SuiteHandler suites(in_data->pica_basicP);
	suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
		"%d == %d", UIP_SortPattern, extra->param_index);*/
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	if (extra->param_index == UIP_UseGPU && params[UIP_UseGPU]->u.bd.value) {
		hasCUDA = callCudaFunc()>0;
		if (!hasCUDA) {
			params[UIP_UseGPU]->ui_flags = PF_PUI_DISABLED;
			params[UIP_UseGPU]->u.bd.value = false;
			params[UIP_UseGPU]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
			suites.ParamUtilsSuite3()->PF_UpdateParamUI(in_data->effect_ref, UIP_UseGPU, params[UIP_UseGPU]);
			suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg, "You don't seem have CUDA, go get a new Nvidia graphics card :))");
		}else {
			suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg, "OK, you have CUDA :)");
		}
	}
	if (extra->param_index == UIP_SortPattern) {
		//change which options to show
		/*params[UIP_RefLayer]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		params[UIP_Angle]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		params[UIP_NumSides]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		params[UIP_CenterPoint]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		params[UIP_CurveAngle]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		params[UIP_WHRatio]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		params[UIP_WaveLength]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		params[UIP_WaveHeight]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		params[UIP_Rotation]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;*/
		switch (params[UIP_SortPattern]->u.pd.value) {
		case PSP_Linear:
			params[UIP_RefLayer]->ui_flags = PF_PUI_DISABLED;
			params[UIP_Angle]->ui_flags = PF_PUI_NONE;
			params[UIP_NumSides]->ui_flags = PF_PUI_DISABLED;
			params[UIP_CenterPoint]->ui_flags = PF_PUI_DISABLED;
			params[UIP_CurveAngle]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WHRatio]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WaveLength]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WaveHeight]->ui_flags = PF_PUI_DISABLED;
			params[UIP_Rotation]->ui_flags = PF_PUI_DISABLED;
			break;
		case PSP_Radial_Spin:
			params[UIP_RefLayer]->ui_flags = PF_PUI_DISABLED;
			params[UIP_Angle]->ui_flags = PF_PUI_DISABLED;
			params[UIP_NumSides]->ui_flags = PF_PUI_DISABLED;
			params[UIP_CenterPoint]->ui_flags = PF_PUI_NONE;
			params[UIP_CurveAngle]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WHRatio]->ui_flags = PF_PUI_NONE;
			params[UIP_WaveLength]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WaveHeight]->ui_flags = PF_PUI_DISABLED;
			params[UIP_Rotation]->ui_flags = PF_PUI_DISABLED;
			break;
		case PSP_Polygon:
			params[UIP_RefLayer]->ui_flags = PF_PUI_DISABLED;
			params[UIP_Angle]->ui_flags = PF_PUI_NONE;
			params[UIP_NumSides]->ui_flags = PF_PUI_NONE;
			params[UIP_CenterPoint]->ui_flags = PF_PUI_NONE;
			params[UIP_CurveAngle]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WHRatio]->ui_flags = PF_PUI_NONE;
			params[UIP_WaveLength]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WaveHeight]->ui_flags = PF_PUI_DISABLED;
			params[UIP_Rotation]->ui_flags = PF_PUI_NONE;
			break;
		case PSP_Spiral:
			params[UIP_RefLayer]->ui_flags = PF_PUI_DISABLED;
			params[UIP_Angle]->ui_flags = PF_PUI_DISABLED;
			params[UIP_NumSides]->ui_flags = PF_PUI_DISABLED;
			params[UIP_CenterPoint]->ui_flags = PF_PUI_NONE;
			params[UIP_CurveAngle]->ui_flags = PF_PUI_NONE;
			params[UIP_WHRatio]->ui_flags = PF_PUI_NONE;
			params[UIP_WaveLength]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WaveHeight]->ui_flags = PF_PUI_DISABLED;
			params[UIP_Rotation]->ui_flags = PF_PUI_NONE;
			break;
		case PSP_Sine:
			params[UIP_RefLayer]->ui_flags = PF_PUI_DISABLED;
			params[UIP_Angle]->ui_flags = PF_PUI_DISABLED;
			params[UIP_NumSides]->ui_flags = PF_PUI_DISABLED;
			params[UIP_CenterPoint]->ui_flags = PF_PUI_DISABLED;
			params[UIP_CurveAngle]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WHRatio]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WaveLength]->ui_flags = PF_PUI_NONE;
			params[UIP_WaveHeight]->ui_flags = PF_PUI_NONE;
			params[UIP_Rotation]->ui_flags = PF_PUI_NONE;
			break;
		case PSP_Triangle:
			params[UIP_RefLayer]->ui_flags = PF_PUI_DISABLED;
			params[UIP_Angle]->ui_flags = PF_PUI_DISABLED;
			params[UIP_NumSides]->ui_flags = PF_PUI_DISABLED;
			params[UIP_CenterPoint]->ui_flags = PF_PUI_DISABLED;
			params[UIP_CurveAngle]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WHRatio]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WaveLength]->ui_flags = PF_PUI_NONE;
			params[UIP_WaveHeight]->ui_flags = PF_PUI_NONE;
			params[UIP_Rotation]->ui_flags = PF_PUI_NONE;
			break;
		case PSP_Saw_Tooth:
			params[UIP_RefLayer]->ui_flags = PF_PUI_DISABLED;
			params[UIP_Angle]->ui_flags = PF_PUI_DISABLED;
			params[UIP_NumSides]->ui_flags = PF_PUI_DISABLED;
			params[UIP_CenterPoint]->ui_flags = PF_PUI_DISABLED;
			params[UIP_CurveAngle]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WHRatio]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WaveLength]->ui_flags = PF_PUI_NONE;
			params[UIP_WaveHeight]->ui_flags = PF_PUI_NONE;
			params[UIP_Rotation]->ui_flags = PF_PUI_NONE;
			break;
		case PSP_Optical_Flow:
			params[UIP_RefLayer]->ui_flags = PF_PUI_NONE;
			params[UIP_Angle]->ui_flags = PF_PUI_DISABLED;
			params[UIP_NumSides]->ui_flags = PF_PUI_DISABLED;
			params[UIP_CenterPoint]->ui_flags = PF_PUI_DISABLED;
			params[UIP_CurveAngle]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WHRatio]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WaveLength]->ui_flags = PF_PUI_DISABLED;
			params[UIP_WaveHeight]->ui_flags = PF_PUI_DISABLED;
			params[UIP_Rotation]->ui_flags = PF_PUI_DISABLED;
			break;
		default:
			break;
		}
		for (int i = UIP_GroupPatternOptionStart; i <= UIP_Rotation; i++) {
			suites.ParamUtilsSuite3()->PF_UpdateParamUI(in_data->effect_ref, i, params[i]);
		}
	}
	return PF_Err_NONE;
}

static PF_Err seqSetup(PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output) {
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	hasCUDA = callCudaFunc()>0;
	return		PF_Err_NONE;
}

DllExport PF_Err 
EntryPointFuncM (
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err		err = PF_Err_NONE;
	
	try {
		switch (cmd) {
			case PF_Cmd_USER_CHANGED_PARAM:
				err = ChangeParam(in_data, out_data, params, output,(PF_UserChangedParamExtra*)extra);
				break;
			case PF_Cmd_ABOUT:
				err = About(in_data, out_data, params, output);
				break;
				
			case PF_Cmd_GLOBAL_SETUP:
				err = GlobalSetup(in_data, out_data, params, output);
				break;
				
			case PF_Cmd_PARAMS_SETUP:
				err = ParamsSetup(in_data, out_data, params, output);
				break;
				
			/*case PF_Cmd_FRAME_SETUP:
				out_data->out_flags |= PF_OutFlag_NOP_RENDER;
				err = PF_Err_NONE;
				break;*/

			case PF_Cmd_UPDATE_PARAMS_UI:
				PF_UserChangedParamExtra temp;
				temp.param_index = UIP_SortPattern;
				err = ChangeParam(in_data, out_data, params, output, &temp);
				break;

			case PF_Cmd_SEQUENCE_SETUP:
				err = seqSetup(in_data,out_data,	params,output);
				break;
			case PF_Cmd_SEQUENCE_RESETUP:
				err = seqSetup(in_data, out_data, params, output);
				break;

			case PF_Cmd_RENDER:
				//TODO: Real render function
				if (hasCUDA && params[UIP_UseGPU]->u.bd.value) {
					//Render in GPU
					AEGP_SuiteHandler suites(in_data->pica_basicP);
					PF_Pixel tempColor;
					tempColor.alpha = (A_u_char)255;
					tempColor.red = (A_u_char)0;
					tempColor.green = (A_u_char)255;
					tempColor.blue = (A_u_char)0;
					suites.FillMatteSuite2()->fill(in_data->effect_ref,&tempColor,&(output->extent_hint), output);
				}else{
					//Fall to CPU Rendering
					AEGP_SuiteHandler suites(in_data->pica_basicP);
					suites.WorldTransformSuite1()->copy_hq(in_data->effect_ref, &params[0]->u.ld, output, NULL, &output->extent_hint);
				}
				break;
		}
	}
	catch(PF_Err &thrown_err){
		err = thrown_err;
	}
	return err;
}
