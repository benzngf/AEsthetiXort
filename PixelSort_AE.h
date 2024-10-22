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

/*
	PixelSort_AE.h
*/

#pragma once

#ifndef AETHESTIXORT_H
#define AETHESTIXORT_H

typedef unsigned char		u_char;
typedef unsigned short		u_short;
typedef unsigned short		u_int16;
typedef unsigned long		u_long;
typedef short int			int16;
#define PF_TABLE_BITS	12
#define PF_TABLE_SZ_16	4096

#define PF_DEEP_COLOR_AWARE 1	// make sure we get 16bpc pixels; 
								// AE_Effect.h checks for this.

#include "AEConfig.h"

#ifdef AE_OS_WIN
	typedef unsigned short PixelType;
	#include <Windows.h>
#endif

#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectCBSuites.h"
#include "String_Utils.h"
#include "AE_GeneralPlug.h"
#include "AEFX_ChannelDepthTpl.h"
#include "AEGP_SuiteHandler.h"

#include "Layer_transformer.h"
#include "PixelSort.h"
/* Versioning information */

#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1



/* Parameter defaults */
	enum UIPARAMS {
		UIP_ThisLayer = 0,
		UIP_GroupThresStart = 1,
		UIP_SortBy,
		UIP_MinThres,
		UIP_MaxThres,
		UIP_GroupThresEnd,
		UIP_RevSortOrder,
		UIP_GroupPatternStart,
		UIP_SortPattern,
		UIP_GroupPatternOptionStart,
		UIP_RefLayer,
		UIP_TimeStep,
		UIP_Angle,
		UIP_NumSides,
		UIP_CenterPoint,
		UIP_CurveAngle,
		UIP_WHRatio,
		UIP_WaveLength,
		UIP_WaveHeight,
		UIP_Rotation,
		UIP_GroupPatternOptionEnd,
		UIP_GroupPatternEnd,
		UIP_AntiAliasing,
		UIP_SortAlpha,
		UIP_UseGPU
	};


#ifdef __cplusplus
	extern "C" {
#endif
PF_Err prepareParams(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], PixelSortPatternParm** &out_patternparam);
PF_Err disposeParams(PF_InData *in_data, PixelSortPatternParm** out_patternparam);
PF_Err prepareGPUinput(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], Pixel** &GPUinputH);
PF_Err copyGPUresult(PF_InData *in_data, PF_OutData *out_data, PF_LayerDef *output, Pixel** GPUinputH);
PF_Err prepareCPUinput(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], Pixel** &CPUinputH);
PF_Err PixelSortCPU(PF_InData	 *in_data, PF_OutData *out_data, PF_ParamDef *params[], PF_LayerDef	*output, PixelSortPatternParm *pattern_parm, Pixel** &CPUinputH);
DllExport	PF_Err 
EntryPointFuncM(	
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra) ;

#ifdef __cplusplus
}
#endif

#endif //PixelSort_AE.h