# -*- coding: utf-8 -*-

"""
***************************************************************************
    Unified_Curb_Layer.py
    ---------------------
    Date                 : February 2021
    Copyright            : (C) 2021, IBI Group
    Email                : prem.thogiti at ibigroup dot com
***************************************************************************
*                                                                         *
*   This program is confidencial and copyright software of IBI Group;     *
*                                                                         *
***************************************************************************
"""

__author__ = 'Prem Kumar Thogiti'
__date__ = 'February 2021'
__copyright__ = '(C) 2021, IBI Group'
import math
import pandas as pd
from qgis.PyQt.QtCore import QVariant
from .pibi import QgsUtils as Qutils
import os
from PyQt5.QtCore import QCoreApplication
import json
import numpy as np
from qgis.core import (QgsApplication,
                        QgsPoint,
                       QgsWkbTypes,
                       QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsDistanceArea,
                       QgsVectorDataProvider,
                       QgsProcessingParameterFileDestination,
                       QgsPointXY,
                       QgsVectorLayer,
                       QgsField,
                       QgsFeature,
                       QgsGeometry,
                       QgsFeatureRequest,
                       edit,
                       QgsCoordinateTransform,
                       QgsFields,
                       QgsVectorFileWriter,
                       QgsCoordinateReferenceSystem,
                       QgsProject,
                       QgsSpatialIndex,
                       QgsProcessingParameterString,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFeatureSource
                       )
import processing

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class GenerateUnifiedCurb(QgsProcessingAlgorithm):
    INPUTGENCURBS = 'INPUTGENCURBS'
    OUTPUT = 'OUTPUT'
    INPUTBYLAWCURBS = 'INPUTBYLAWCURBS'
    INPUTROADS = 'INPUTROADS'
    DEFAULT_BYLAW = 'DEFAULT_BYLAW'
    MIN_REGULATION_LENGTH = 'MIN_REGULATION_LENGTH'   #Threshold length of regulation or undefined segment

    def group(self):
        return self.tr('CurbIQ Tools')

    def groupId(self):
        return 'ibitools'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return GenerateUnifiedCurb()

    def __init__(self):
        super().__init__()
        self.bylaw_tr = None

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUTBYLAWCURBS,
                self.tr('Input Bylaw curb layer'),
                [QgsProcessing.TypeVectorLine]
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUTGENCURBS,
                self.tr('Input General Curb layer'),
                [QgsProcessing.TypeVectorLine]
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUTROADS,
                self.tr('Input Road centerline layer'),
                [QgsProcessing.TypeVectorLine]
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.DEFAULT_BYLAW,
                self.tr('Input default bylawid and reason (saparated by ;)'),
                defaultValue="90001;Parking"
            )
        )

        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT, self.tr('Unified Curb Layer'),
                fileFilter="*.geojson"
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.MIN_REGULATION_LENGTH,
                description=self.tr('MIN_REGULATION_LENGTH'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
                optional=False
            )
        )

    def name(self):
        return 'GenerateUnifiedCurb'

    def displayName(self):
        return self.tr('IBI - Generate Unified Curb')

    def get_azimuth_direction(self,bearing):
        if bearing < 45:
            return "NORTH"
        if 45 <= bearing < 135:
            return "EAST"
        if 135 <= bearing < 225:
            return "SOUTH"
        if 225 <= bearing < 315:
            return "WEST"
        if bearing >= 315:
            return "NORTH"

    def get_opposite_direction(self, direction):
        if direction.upper() == 'SOUTH':
            return 'NORTH'
        if direction.upper() == 'WEST':
            return 'EAST'
        if direction.upper() == 'NORTH':
            return 'SOUTH'
        if direction.upper() == 'EAST':
            return 'WEST'
        if direction.upper() == 'UNKNOWN':
            return 'UNKNOWN'

    def get_xypoint(self, point_geom):
        x = y = 0
        if isinstance(point_geom, QgsPoint):
            pt = QgsPointXY(point_geom)
            return pt
        elif isinstance(point_geom, QgsPointXY):
            return point_geom
        elif isinstance(point_geom, QgsGeometry):
            pt = QgsPointXY(point_geom.asPoint())
            return pt
    def processAlgorithm(self, parameters, context, feedback):
        number_of_bylaws=60
        in_obj_gencurbs = self.parameterAsVectorLayer(parameters, self.INPUTGENCURBS, context)
        in_obj_bylawcurbs = self.parameterAsVectorLayer(parameters, self.INPUTBYLAWCURBS, context)
        in_obj_roads = self.parameterAsVectorLayer(parameters, self.INPUTROADS, context)
        min_regulation_len = self.parameterAsDouble(parameters, self.MIN_REGULATION_LENGTH, context)
        default_bylaws = self.parameterAsString(parameters, self.DEFAULT_BYLAW, context)
        common_bylaw = default_bylaws.split(';')[0]  # 90000
        common_reason = default_bylaws.split(';')[1]  # "Free Parking"
        additional_bylaws = {float(common_bylaw): common_reason}

        if in_obj_bylawcurbs is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUTBYLAWCURBS))
        if in_obj_gencurbs is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUTGENCURBS))
        if in_obj_roads is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUTROADS))
        app_crs = 'EPSG:26917'
        out_crs = 'EPSG:4326'
        gen_crs = in_obj_gencurbs.sourceCrs()
        curb_crs = in_obj_bylawcurbs.sourceCrs()
        roads_crs = in_obj_roads.sourceCrs()
        bylaw_tr = None
        if gen_crs.authid() != app_crs:
            bylaw_tr = QgsCoordinateTransform(QgsCoordinateReferenceSystem(app_crs),
                                              QgsCoordinateReferenceSystem(out_crs),
                                              QgsProject.instance())
            parameter = {'INPUT': in_obj_gencurbs, 'TARGET_CRS': app_crs,
                         'OUTPUT': 'memory:'}

            algresult = processing.run('native:reprojectlayer', parameter, feedback=feedback)
            in_obj_gencurbs = algresult['OUTPUT']

        if curb_crs.authid() != app_crs:
            parameter = {'INPUT': in_obj_bylawcurbs, 'TARGET_CRS': app_crs,
                         'OUTPUT': 'memory:'}

            algresult = processing.run('native:reprojectlayer', parameter, feedback=feedback)
            in_obj_bylawcurbs = algresult['OUTPUT']

        if roads_crs.authid() != app_crs:
            parameter = {'INPUT': in_obj_roads, 'TARGET_CRS': app_crs,
                         'OUTPUT': 'memory:'}

            algresult = processing.run('native:reprojectlayer', parameter, feedback=feedback)
            in_obj_roads = algresult['OUTPUT']

        """Generates general curb layer for the given area of interest or city boundary !"""
        output_filepath = self.parameterAsFileOutput(
            parameters,
            self.OUTPUT,
            context)
        feedback.pushInfo("Executing Algorithm!!!")
        total_subprocesses = 9 #Update when a new sub processing is added.
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=Warning)
        app_crs = 'EPSG:26917'

        fields = [f.name().upper() for f in in_obj_gencurbs.fields() if f.name().upper() == "UID"]
        if len(fields) == 0:
            caps = in_obj_gencurbs.dataProvider().capabilities()
            if caps & QgsVectorDataProvider.AddAttributes:
                res = in_obj_gencurbs.dataProvider().addAttributes([QgsField("UID", QVariant.String)])
                in_obj_gencurbs.updateFields()

        al_dict = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H'}
        ulist = {}
        gen_id_uid = {}
        idx_gen_feats = QgsSpatialIndex(QgsSpatialIndex.FlagStoreFeatureGeometries)
        uid_street_dict = {}
        gen_curb_geoms = {}
        total_count = 100.0 / in_obj_gencurbs.featureCount() if in_obj_gencurbs.featureCount() else 0
        print("Sub-process - Adding UIDs to general curb segments - 1/{}".format(total_subprocesses))
        all_feats = in_obj_gencurbs.getFeatures()
        all_feats_out = []
        for current, urow in enumerate(all_feats):
            if feedback.isCanceled():
                break
            feedback.setProgress(int(current * total_count))
            if urow["uc_id"] not in ulist.keys():
                urow["UID"] = str(urow["uc_id"]) + al_dict[1]
                ulist[urow["uc_id"]] = 1
                gen_id_uid[urow.id()] = str(urow["uc_id"]) + al_dict[1]
            elif urow["uc_id"] in ulist:
                ulist[urow["uc_id"]] = ulist[urow["uc_id"]] + 1
                urow["UID"] = str(urow["uc_id"]) + al_dict[ulist[urow["uc_id"]]]
                gen_id_uid[urow.id()] = str(urow["uc_id"]) + al_dict[ulist[urow["uc_id"]]]
            all_feats_out.append(urow)

            # Add feature to the spatial index.
            idx_gen_feats.addFeature(urow)

            # Create dictionary
            uid_street_dict[urow["UID"]] = (
            urow["osm_id"], urow["FClass"], urow["streetName"], urow["OneWay"], urow["MaxSpeed"], urow["sideOfStreet"],
            urow["uc_id"])

            # Create dictionary of geoms
            gen_curb_geoms[urow["UID"]] = urow.geometry()

        fields = [f.name().upper() for f in in_obj_bylawcurbs.fields() if f.name().upper() == "UID"]
        if len(fields) == 0:
            caps = in_obj_bylawcurbs.dataProvider().capabilities()
            if caps & QgsVectorDataProvider.AddAttributes:
                res = in_obj_bylawcurbs.dataProvider().addAttributes([QgsField("UID", QVariant.String)])
                in_obj_bylawcurbs.updateFields()

        ulist = {}
        idsNotFoundGenSegments = []
        bylaw_UID_gen_UID_comb = {}
        total_count = 100.0 / in_obj_bylawcurbs.featureCount() if in_obj_bylawcurbs.featureCount() else 0
        print("Sub-process - Adding UIDs to bylaw curb segments - 2/{}".format(total_subprocesses))
        all_feats = in_obj_bylawcurbs.getFeatures()
        # with edit(in_obj_bylawcurbs):
        all_feats_bylaws = []
        for current, feat in enumerate(all_feats):
            if feedback.isCanceled():
                break
            feedback.setProgress(int(current * total_count))
            if feat["BYLAWID"] not in ulist.keys():
                feat["UID"] = str(feat["BYLAWID"]) + '_1'
                ulist[feat["BYLAWID"]] = 1
            elif feat["BYLAWID"] in ulist:
                ulist[feat["BYLAWID"]] = ulist[feat["BYLAWID"]] + 1
                feat["UID"] = str(feat["BYLAWID"]) + '_{}'.format(str(ulist[feat["BYLAWID"]]))
            all_feats_bylaws.append(feat)

            bylaw_midpt = (feat.geometry()).interpolate(
                feat.geometry().length() / 2.0)  # used to find a point at a specified distance (half) along a LineString
            intersecting_genIds = idx_gen_feats.intersects(bylaw_midpt.buffer(3, 5).boundingBox())
            nearest_gen_geom = None
            max_len = 9999
            nearest_gen_id = None
            for gen_id in intersecting_genIds:
                res = idx_gen_feats.geometry(gen_id).closestSegmentWithContext(bylaw_midpt.asPoint())
                if res[0] < max_len:
                    max_len = res[0]
                    nearest_gen_geom = idx_gen_feats.geometry(gen_id).convertToType(QgsWkbTypes.LineGeometry, False)
                    nearest_gen_id = gen_id
            if not nearest_gen_geom:
                idsNotFoundGenSegments.append(feat.id())
                continue
            bylaw_UID_gen_UID_comb[feat['UID']] = gen_id_uid[
                nearest_gen_id]  # bylaw_UID_gen_UID_comb={ByLaw UID:Gen Curb UID}

        unique_base_uids = list(set(bylaw_UID_gen_UID_comb.values()))  # unique gen curb UID's

        final_dict = {}
        icount = 0

        total_count = 100.0 / in_obj_gencurbs.featureCount() if in_obj_gencurbs.featureCount() else 0
        print("Sub-process - Collecting all vertices from general curb - 3/{}".format(total_subprocesses))
        for current, srow in enumerate(all_feats_out):
            if feedback.isCanceled():
                break
            feedback.setProgress(int(current * total_count))
            if str(srow["UID"]) not in unique_base_uids:
                continue
            for part in srow.geometry().constGet():
                l = len(part)
                for idx, pt in enumerate(part):
                    seedmidpt_row = {'UID': None, 'OSM_ID': None, 'TYPE': None, 'BYLAWID': None, 'VORDER': None,
                                     'LINEAGE': None, 'geom': None, 'SHAPE_X': None, 'SHAPE_Y': None}
                    seedmidpt_row['geom'] = pt
                    seedmidpt_row['SHAPE_X'] = pt.x()
                    seedmidpt_row['SHAPE_Y'] = pt.y()
                    seedmidpt_row['OSM_ID'] = srow["OSM_ID"]
                    seedmidpt_row['UID'] = srow["UID"]
                    seedmidpt_row['BYLAWID'] = -1  # -1 will be assigned for the starting or ending of the gen curb
                    if idx == 0:
                        seedmidpt_row["TYPE"] = "START"
                        seedmidpt_row["VORDER"] = 0  # vertices order
                        final_dict[icount] = seedmidpt_row
                        icount = icount + 1
                    elif idx == l - 1:
                        seedmidpt_row["TYPE"] = "END"
                        seedmidpt_row["VORDER"] = 99999
                        final_dict[icount] = seedmidpt_row
                        icount = icount + 1
                    else:
                        seedmidpt_row["TYPE"] = "VERTEX"
                        seedmidpt_row["VORDER"] = idx
                        final_dict[icount] = seedmidpt_row
                        icount = icount + 1

        total_count = 100.0 / in_obj_bylawcurbs.featureCount() if in_obj_bylawcurbs.featureCount() else 0
        feedback.pushInfo("Sub-process - Collecting start and end points from Bylaw segments - 4/{}".format(total_subprocesses))
        for current, srow in enumerate(all_feats_bylaws):
            bylaw_seg = srow.geometry()
            if "Multi" not in bylaw_seg.asWkt():
                bylaw_seg = bylaw_seg.convertToType(QgsWkbTypes.LineGeometry, True)
            for part in bylaw_seg.constGet():
                seedmidpt_row = {'UID': None, 'OSM_ID': None, 'TYPE': None, 'BYLAWID': None, 'VORDER': None,
                                 'LINEAGE': None, 'geom': None, 'SHAPE_X': None, 'SHAPE_Y': None}
                seedmidpt_row['geom'] = QgsGeometry.fromPointXY(QgsPointXY(part[0]))
                seedmidpt_row['SHAPE_X'] = part[0].x()
                seedmidpt_row['SHAPE_Y'] = part[0].y()
                if srow["UID"] in bylaw_UID_gen_UID_comb.keys(): seedmidpt_row["OSM_ID"] = bylaw_UID_gen_UID_comb[
                    srow["UID"]]
                seedmidpt_row["UID"] = srow["UID"]
                seedmidpt_row["TYPE"] = "START"
                seedmidpt_row["BYLAWID"] = srow["BYLAWID"]
                seedmidpt_row["VORDER"] = 0
                final_dict[icount] = seedmidpt_row
                icount = icount + 1

                seedmidpt_row = {'UID': None, 'OSM_ID': None, 'TYPE': None, 'BYLAWID': None, 'VORDER': None,
                                 'LINEAGE': None, 'geom': None, 'SHAPE_X': None, 'SHAPE_Y': None}
                seedmidpt_row['geom'] = QgsGeometry.fromPointXY(QgsPointXY(part[-1]))
                seedmidpt_row['SHAPE_X'] = part[-1].x()
                seedmidpt_row['SHAPE_Y'] = part[-1].y()
                if srow["UID"] in bylaw_UID_gen_UID_comb.keys(): seedmidpt_row["OSM_ID"] = bylaw_UID_gen_UID_comb[
                    srow["UID"]]
                seedmidpt_row["UID"] = srow["UID"]
                seedmidpt_row["TYPE"] = "END"
                seedmidpt_row["BYLAWID"] = srow["BYLAWID"]
                seedmidpt_row["VORDER"] = 99999

                final_dict[icount] = seedmidpt_row
                icount = icount + 1

        # update LINEAGE, SHAPE_X, SHAPE_Y fields
        bylaw_endpts = pd.DataFrame.from_dict(final_dict, "index")
        bylaw_endpts.index = bylaw_endpts.index.astype(
            int)  # converts the index of the DataFrame df_uni_df to integers.

        cnt = 0.0
        reorder_dict = {}

        total_count = 100.0 / len(unique_base_uids)
        feedback.pushInfo(
            "Sub-process - Calculating lineages - 5/{}".format(total_subprocesses))
        for current, item in enumerate(unique_base_uids):
            if feedback.isCanceled():
                break
            feedback.setProgress(int(current * total_count))
            cnt = cnt + 1
            start_geom = gen_curb_geoms[item]
            pd_frame_q = bylaw_endpts.query("(UID=='{}') or (OSM_ID=='{}')".format(item, item))
            for index, row in pd_frame_q.iterrows():
                closeSegResult = start_geom.closestSegmentWithContext(self.get_xypoint(row[
                                                                                      'geom']))  # (sq dis btw i/p pt and projected point, The clost pt on the segment to the input point, index of)
                nearest_pt = QgsGeometry.fromPointXY(closeSegResult[
                                                         1])  # PROJECTED POINT in gen curb corresponding to bylaw end point or gen curb start end or vertex
                l_calc = Qutils.measureOnLine_new(start_geom,
                                                  nearest_pt)  # distance from the starting of the gen curb to the projected point
                if str(l_calc) != 'None':
                    bylaw_endpts.at[index, 'LINEAGE'] = float(l_calc)
                    pnt = None
                    if isinstance(nearest_pt, QgsPoint):
                        pnt = nearest_pt
                    elif isinstance(nearest_pt, QgsGeometry):
                        pnt = nearest_pt.asPoint()

                    # update shape_x and shape_y field values of bylaw from general curb nearest point
                    bylaw_endpts.at[index, 'SHAPE_X'] = pnt.x()
                    bylaw_endpts.at[index, 'SHAPE_Y'] = pnt.y()

                    if row['BYLAWID'] in ('-1.0', '-1', -1):  # gen curb
                        continue
                    if row['BYLAWID'] not in reorder_dict.keys():
                        reorder_dict[row['BYLAWID']] = {row['TYPE']: float(l_calc)}
                    elif row['BYLAWID'] in reorder_dict.keys():
                        t_dict = reorder_dict[row['BYLAWID']]
                        t_dict[row['TYPE']] = float(l_calc)
                        reorder_dict[row['BYLAWID']] = t_dict

        bylaws_need_reorder = []
        print("Finalize reorder dictionary !")
        for key, value in reorder_dict.items():
            if not value['START'] and not value['END']:
                print("bylaws had null {}".format(key))
                continue
            if value['START'] > value['END']:
                bylaws_need_reorder.append(key)

        total_count = 100.0 / len(bylaw_endpts)
        feedback.pushInfo(
            "Sub-process - Updating reorder information - 7/{}".format(total_subprocesses))
        for current, pt in bylaw_endpts.iterrows():
            if feedback.isCanceled():
                break
            feedback.setProgress(int(current * total_count))
            if pt['BYLAWID'] in bylaws_need_reorder:
                bylaw_endpts.at[current, 'TYPE'] = 'END' if pt['TYPE'] == 'START' else 'START'
                bylaw_endpts.at[current, 'VORDER'] = 0 if float(pt['VORDER']) == 99999 else 99999

        active_geoms = {}
        base_curb_geoms = {}
        cnt = 0.0
        total_count = 100.0 / len(unique_base_uids)
        feedback.pushInfo(
            "Sub-process - MAIN segments creation - 8/{}".format(total_subprocesses))
        for current, item in enumerate(unique_base_uids):
            if feedback.isCanceled():
                break
            feedback.setProgress(int(current * total_count))
            internal_active_geoms = []
            internal_base_geoms = []
            cnt = cnt + 1
            df = bylaw_endpts.query("UID=='{}' or OSM_ID=='{}'".format(item, item))
            df = df.sort_values(by=["LINEAGE"], ascending=(True))

            # sliver geometry correction in the starting and ending of a road segment
            df.reset_index(drop=True, inplace=True)

            last_bylaw_row = df.loc[(df['TYPE'] == 'END') & (df['BYLAWID'] != -1)].tail(1)
            first_gen_curb_row = df.loc[(df['TYPE'] == 'START') & (df['BYLAWID'] == -1)].head(1)
            last_gen_curb_row = df.loc[(df['TYPE'] == 'END') & (df['BYLAWID'] == -1)].tail(1)

            first_gen_curb_lineage = first_gen_curb_row['LINEAGE'].values[0]
            first_gen_curb_x_shape = first_gen_curb_row['SHAPE_X'].values[0]
            first_gen_curb_y_shape = first_gen_curb_row['SHAPE_Y'].values[0]
            last_gen_curb_x_shape = last_gen_curb_row['SHAPE_X'].values[0]
            last_gen_curb_y_shape = last_gen_curb_row['SHAPE_Y'].values[0]
            last_gen_curb_lineage = last_gen_curb_row['LINEAGE'].values[0]

            start_geom = gen_curb_geoms[item]

            start_df = df[(df['LINEAGE'] == 0.0) | ((df['LINEAGE'] > 0.0) & (df['LINEAGE'] < min_regulation_len))]
            last_index = len(start_df) - 1
            if first_gen_curb_x_shape is not None and first_gen_curb_y_shape is not None and not start_df.empty:
                if not all(start_df['SHAPE_X'] == first_gen_curb_x_shape):
                    df.loc[:last_index, 'SHAPE_X'] = first_gen_curb_x_shape
                if not all(start_df['SHAPE_Y'] == first_gen_curb_y_shape):
                    df.loc[:last_index, 'SHAPE_Y'] = first_gen_curb_y_shape
                if not all(start_df['LINEAGE'] == first_gen_curb_lineage):
                    df.loc[:last_index, 'LINEAGE'] = first_gen_curb_lineage

            end_df = df[((last_gen_curb_lineage - df['LINEAGE']) > 0) & (
                        (last_gen_curb_lineage - df['LINEAGE']) < min_regulation_len)]
            no_of_indices = (len(end_df)) + 1
            sel_indices_from_last = df.tail(no_of_indices).index
            if last_gen_curb_x_shape is not None and last_gen_curb_y_shape is not None and not end_df.empty:
                if not all(end_df['SHAPE_X'] == last_gen_curb_x_shape):
                    df.loc[sel_indices_from_last, 'SHAPE_X'] = last_gen_curb_x_shape
                if not all(end_df['SHAPE_Y'] == last_gen_curb_y_shape):
                    df.loc[sel_indices_from_last, 'SHAPE_Y'] = last_gen_curb_y_shape
                if not all(end_df['LINEAGE'] == last_gen_curb_lineage):
                    df.loc[sel_indices_from_last, 'LINEAGE'] = last_gen_curb_lineage

            # Sliver geometry correction in the middle of any road segments
            result_dict = {}
            # Iterate over the dataframe indices
            for i in range(len(df) - 1):
                if i not in start_df.index and i not in end_df.index and i not in result_dict:
                    current_lineage = df.at[i, 'LINEAGE']
                    next_lineage = df.at[i + 1, 'LINEAGE']
                    current_typ = df.at[i, 'TYPE']
                    next_typ = df.at[i + 1, 'TYPE']

                    if 0 < (next_lineage - current_lineage) < min_regulation_len and (
                            current_typ != 'VERTEX' or next_typ != 'VERTEX'):
                        consecutive_indices = [i]
                        consecutive_lineages = [current_lineage]
                        consecutive_shape_x = [df.at[i, 'SHAPE_X']]
                        consecutive_shape_y = [df.at[i, 'SHAPE_Y']]

                        # Find consecutive rows with the 'LINEAGE' difference between 0 and 1
                        j = i + 1
                        while j < len(df) and 0 <= (
                                df.at[j, 'LINEAGE'] - df.at[j - 1, 'LINEAGE']) < min_regulation_len and (
                                df.loc[j, 'TYPE'] != 'VERTEX' or df.loc[j - 1, 'TYPE'] != 'VERTEX'):
                            consecutive_indices.append(j)
                            consecutive_lineages.append(df.at[j, 'LINEAGE'])
                            consecutive_shape_x.append(df.at[j, 'SHAPE_X'])
                            consecutive_shape_y.append(df.at[j, 'SHAPE_Y'])
                            j += 1

                        avg_lineage = sum(consecutive_lineages) / len(consecutive_lineages)
                        avg_shape_x = sum(consecutive_shape_x) / len(consecutive_shape_x)
                        avg_shape_y = sum(consecutive_shape_y) / len(consecutive_shape_y)

                        for idx in consecutive_indices:
                            result_dict[idx] = {'LINEAGE': avg_lineage, 'SHAPE_X': avg_shape_x, 'SHAPE_Y': avg_shape_y}

            for idx, values in result_dict.items():
                df.at[idx, 'SHAPE_X'] = values['SHAPE_X']
                df.at[idx, 'SHAPE_Y'] = values['SHAPE_Y']
                df.at[idx, 'LINEAGE'] = values['LINEAGE']

            active_regulations = []
            ln_array = []
            line_started = False
            last_lineage = 0.0
            last_key = ''
            for row_index, ind in df.iterrows():
                bylaw = str(df.at[row_index, 'BYLAWID'])

                if bylaw in ('-1.0', '-1') and df.at[row_index, 'TYPE'] == 'START':  # gen curb start
                    last_lineage = df.at[row_index, 'LINEAGE']
                    ln_array.append(QgsPointXY(df.at[row_index, 'SHAPE_X'], df.at[row_index, 'SHAPE_Y']))
                    active_regulations.append(bylaw)
                    line_started = True
                    continue

                if bylaw in ('-1.0', '-1') and df.at[row_index, 'TYPE'] == 'END':  # gen curb end
                    last_key = "{}".format(','.join(set(active_regulations)))
                    ln_array.append(QgsPointXY(df.at[row_index, 'SHAPE_X'], df.at[row_index, 'SHAPE_Y']))
                    last_lineage = df.at[row_index, 'LINEAGE']
                    if last_key in ('-1.0', '-1'):
                        internal_base_geoms.append(QgsGeometry.fromPolylineXY(ln_array))
                    else:
                        internal_active_geoms.append((last_key, QgsGeometry.fromPolylineXY(ln_array)))
                    ln_array = []
                    ln_array.append(QgsPointXY(df.at[row_index, 'SHAPE_X'], df.at[row_index, 'SHAPE_Y']))
                    break

                if bylaw in ('-1.0', '-1') and df.at[row_index, 'TYPE'] == 'VERTEX' and abs(
                        last_lineage - df.at[row_index, 'LINEAGE']) > 0.2:  # gen curb vertex
                    last_lineage = df.at[row_index, 'LINEAGE']
                    ln_array.append(QgsPointXY(df.at[row_index, 'SHAPE_X'], df.at[row_index, 'SHAPE_Y']))
                    continue

                if bylaw not in ('-1.0', '-1') and bylaw not in active_regulations:  # bylaw start
                    last_key = "{}".format(','.join(set(active_regulations)))
                    ln_array.append(QgsPointXY(df.at[row_index, 'SHAPE_X'], df.at[row_index, 'SHAPE_Y']))
                    last_lineage = df.at[row_index, 'LINEAGE']
                    if last_key in ('-1.0', '-1'):
                        internal_base_geoms.append(QgsGeometry.fromPolylineXY(ln_array))
                    else:
                        internal_active_geoms.append((last_key, QgsGeometry.fromPolylineXY(ln_array)))
                    ln_array = []
                    ln_array.append(QgsPointXY(df.at[row_index, 'SHAPE_X'], df.at[row_index, 'SHAPE_Y']))
                    active_regulations.append(bylaw)
                    continue

                if bylaw not in ('-1.0', '-1') and bylaw in active_regulations:  # bylaw end
                    last_key = "{}".format(','.join(set(active_regulations)))
                    ln_array.append(QgsPointXY(df.at[row_index, 'SHAPE_X'], df.at[row_index, 'SHAPE_Y']))
                    last_lineage = df.at[row_index, 'LINEAGE']
                    if last_key in ('-1.0', '-1'):
                        internal_base_geoms.append(QgsGeometry.fromPolylineXY(ln_array))
                    else:
                        internal_active_geoms.append((last_key, QgsGeometry.fromPolylineXY(ln_array)))
                    ln_array = []
                    ln_array.append(QgsPointXY(df.at[row_index, 'SHAPE_X'], df.at[row_index, 'SHAPE_Y']))
                    active_regulations = [s for s in active_regulations if s != bylaw]
                    continue
            # all geoms except geometries where only gen curb is there
            active_geoms[item] = internal_active_geoms
            # geometries corresponding to the segment in a particular gen curb uid where only gen curb is there no regulations
            base_curb_geoms[item] = internal_base_geoms

        unified_fields = QgsFields()
        unified_fields.append(QgsField("segmentID", QVariant.Int))
        unified_fields.append(QgsField("streetName", QVariant.String))
        for i in range(1, number_of_bylaws + 1):
            unified_fields.append(QgsField("P-{}".format(i), QVariant.Double))
            unified_fields.append(QgsField("P-{}R".format(i), QVariant.String))
        unified_fields.append(QgsField("baysCount", QVariant.Int))
        unified_fields.append(QgsField("baysAngle", QVariant.String))
        unified_fields.append(QgsField("sideOfStreet", QVariant.String))
        unified_fields.append(QgsField("LocStart", QVariant.String))
        unified_fields.append(QgsField("LocEnd", QVariant.String))
        unified_fields.append(QgsField("shstRefId", QVariant.String))
        unified_fields.append(QgsField("osm_id", QVariant.Int))
        unified_fields.append(QgsField("uc_id", QVariant.Int))
        unified_fields.append(QgsField("FClass", QVariant.String))
        unified_fields.append(QgsField("OneWay", QVariant.String))
        unified_fields.append(QgsField("MaxSpeed", QVariant.Int))
        unified_fields.append(QgsField("BYLAWS", QVariant.String))
        unified_fields.append(QgsField("BASE_C_ID", QVariant.String))
        unified_fields.append(QgsField("from_node", QVariant.Int))
        unified_fields.append(QgsField("to_node", QVariant.Int))

        unified_feats = []
        for key, value in active_geoms.items():
            (osm_id, FClass, streetName, OneWay, MaxSpeed, sideOfStreet, uc_id) = uid_street_dict[
                key] if key in uid_street_dict.keys() else ("", "", "", "")
            for (key1, value1) in value:
                uni_feat = QgsFeature()
                uni_feat.setFields(unified_fields)
                uni_feat.setGeometry(value1)
                uni_feat["BYLAWS"] = key1
                uni_feat["streetName"] = streetName
                uni_feat["osm_id"] = osm_id
                uni_feat["FClass"] = FClass
                uni_feat["OneWay"] = OneWay
                uni_feat["MaxSpeed"] = MaxSpeed
                uni_feat["sideOfStreet"] = sideOfStreet
                uni_feat["uc_id"] = uc_id
                uni_feat["BASE_C_ID"] = key
                unified_feats.append(uni_feat)
        for key2, value2 in base_curb_geoms.items():
            if str(key2) != 'None' and key2 in uid_street_dict.keys():
                (osm_id, FClass, streetName, OneWay, MaxSpeed, sideOfStreet, uc_id) = uid_street_dict[key2]
            else:
                osm_id = ''
                FClass = ''
                streetName = ''
                OneWay = ''
                MaxSpeed = 0
                sideOfStreet = ''
                uc_id = 0
            for item in value2:
                uni_feat = QgsFeature()
                uni_feat.setFields(unified_fields)
                uni_feat.setGeometry(item)
                uni_feat["BYLAWS"] = '-1.0'
                uni_feat["streetName"] = streetName
                uni_feat["osm_id"] = osm_id
                uni_feat["FClass"] = FClass
                uni_feat["OneWay"] = OneWay
                uni_feat["MaxSpeed"] = MaxSpeed
                uni_feat["sideOfStreet"] = sideOfStreet
                uni_feat["uc_id"] = uc_id
                uni_feat["BASE_C_ID"] = key2
                unified_feats.append(uni_feat)
        ic = 0
        d = {}
        for entry in all_feats_bylaws:
            # add a dictionary entry to the final dictionary
            d[ic] = {"ByLawID": entry['ByLawID'],
                     "priority": entry['priority'],
                     "Reason": entry['Reason']}
            ic = ic + 1

        ## updating the next field as default bylaw id and reason, after the least priority bylaw id and reason
        bylaws = pd.DataFrame.from_dict(d, "index")
        bylaws.index = bylaws.index.astype(int)
        for urow in unified_feats:
            if str(urow["BYLAWS"]) == 'None' or urow["BYLAWS"] == "":
                continue
            ins = urow["BYLAWS"]
            ins = ins.split(',')
            ins = [float(x) for x in ins]
            where_cl = "ByLawID == @ins"  # The condition checks if the ByLawID field is equal to any of the values in the ins list.
            total_bylaws = 0
            trackids = []
            bylaws_q = bylaws.query(where_cl)
            bylaws_q = bylaws_q.sort_values('priority')
            for index, ind in bylaws_q.iterrows():
                if str(ind['ByLawID']) in trackids:
                    continue
                total_bylaws = total_bylaws + 1
                urow["P-{}".format(total_bylaws)] = str(ind['ByLawID'])
                urow["P-{}R".format(total_bylaws)] = ind['Reason']
                trackids.append(str(ind['ByLawID']))
            if total_bylaws < 8:  # maximum overlap of 8 bylaws has observed
                for k, v in additional_bylaws.items():
                    if total_bylaws < 8:
                        total_bylaws = total_bylaws + 1
                        urow["P-{}".format(total_bylaws)] = float(k)
                        urow["P-{}R".format(total_bylaws)] = v
            else:
                print(urow["BYLAWS"])

        #####ENABLE    # updating "P-1" and "P-1R" fields of those features which contains no bylaws (non active geometries)
        base_curb_uids = [row["BASE_C_ID"] for row in unified_feats]  # gen curb uc id s for active geometries
        for genf in all_feats_out:
            if genf['UID'] not in base_curb_uids:
                gen_feat = QgsFeature()
                gen_feat.setFields(unified_fields)
                gen_feat.setGeometry(genf.geometry())
                gen_feat["BYLAWS"] = '-1'
                gen_feat["streetName"] = genf['streetName']
                gen_feat["osm_id"] = genf['osm_id']
                gen_feat["uc_id"] = genf['uc_id']
                gen_feat["FClass"] = genf['FClass']
                gen_feat["OneWay"] = genf['OneWay']
                gen_feat["MaxSpeed"] = genf['MaxSpeed']
                gen_feat["BASE_C_ID"] = genf["UID"]
                for k, v in additional_bylaws.items():
                    gen_feat["P-1"] = float(k)
                    gen_feat["P-1R"] = v
                unified_feats.append(gen_feat)

        unified_feats = [feat for feat in unified_feats if feat.geometry().length() > 1.0]

        ############# FROM TO NODE POPULATION ####################      side of street and street name population
        idx_gen_feats = QgsSpatialIndex(QgsSpatialIndex.FlagStoreFeatureGeometries)
        idx_gen_feats.addFeatures(in_obj_roads.getFeatures())
        distance1 = QgsDistanceArea()
        name_dict = {r.id(): r['name'] for r in in_obj_roads.getFeatures()}
        total_count = 100.0 / len(unified_feats)
        print("Sub-process - Calculating sideOfStreet and StreetName - 9/{}".format(total_subprocesses))
        for current, row in enumerate(unified_feats):
            if feedback.isCanceled():
                break
            feedback.setProgress(int(current*total_count))
            geom = row.geometry()
            if "Multi" not in row.geometry().asWkt():
                geom = geom.convertToType(QgsWkbTypes.LineGeometry, True)
            for part in geom.constGet():
                c_b = math.degrees(distance1.bearing(QgsPointXY(part[0].x(), part[0].y()),
                                                     QgsPointXY(QgsPointXY(part[-1].x(), part[-1].y()))))
                if c_b < 0.0:  # if you want all positive angles
                    c_b += 360
                c_dir = self.get_azimuth_direction(c_b)
                midpoint = (row.geometry()).interpolate(
                    row.geometry().length() / 2.0)  # calculating the midpoint of unified feat
                intersecting_genIds = idx_gen_feats.intersects(midpoint.buffer(25, 5).boundingBox())

                nearest_gen_geom = None
                max_len = 9999
                nearest_gen_id = None
                for gen_id in intersecting_genIds:
                    res = idx_gen_feats.geometry(gen_id).closestSegmentWithContext(midpoint.asPoint())
                    # closestSegmentWithContext(self, point: QgsPointXY, epsilon: float = DEFAULT_SEGMENT_EPSILON)→ Tuple[float, QgsPointXY, int, int]
                    if res[0] < max_len:
                        for part in idx_gen_feats.geometry(gen_id).constGet():
                            start_point = QgsGeometry.fromPointXY(QgsPointXY(part[0].x(), part[0].y()))
                            end_point = QgsGeometry.fromPointXY(QgsPointXY(part[-1].x(), part[-1].y()))
                            f_bearing = math.degrees(distance1.bearing(start_point.asPoint(),
                                                                       end_point.asPoint()))
                            if f_bearing < 0.0:  # if you want all positive angles
                                f_bearing += 360
                            if self.get_azimuth_direction(f_bearing) in (c_dir, self.get_opposite_direction((c_dir))):
                                max_len = res[0]
                                nearest_gen_geom = idx_gen_feats.geometry(gen_id).convertToType(
                                    QgsWkbTypes.LineGeometry, False)
                                nearest_gen_id = gen_id

                if not nearest_gen_geom:
                    for gen_id in intersecting_genIds:
                        res = idx_gen_feats.geometry(gen_id).closestSegmentWithContext(midpoint.asPoint())
                        if res[0] < max_len:
                            max_len = res[0]
                            nearest_gen_geom = idx_gen_feats.geometry(gen_id).convertToType(QgsWkbTypes.LineGeometry,
                                                                                            False)
                            nearest_gen_id = gen_id

                if not nearest_gen_geom:
                    print(row['segmentID'])
                    continue
                closeSegResult = nearest_gen_geom.closestSegmentWithContext(midpoint.asPoint())
                row['sideOfStreet'] = "LEFT" if closeSegResult[3] < 0 else "RIGHT"
                if name_dict[nearest_gen_id]:
                    row['streetName'] = name_dict[nearest_gen_id].upper()

        #####ENABLE    #update 'segment id' and 'baysCount' fields
        if bylaw_tr:
            for i, item in enumerate(unified_feats):
                output_geometry = item.geometry()
                output_geometry.transform(bylaw_tr)
                item['segmentID'] = i + 1
                item['baysCount'] = round(item.geometry().length() / 8, 0)
                item.setGeometry(output_geometry)

        file_name, file_extension = os.path.splitext(output_filepath)
        if file_extension.upper() != ".GEOJSON":
            output_filepath = output_filepath + '.geojson'
        print(output_filepath)
        writer = QgsVectorFileWriter(output_filepath, 'UTF-8', unified_fields, in_obj_gencurbs.wkbType(),
                                     QgsCoordinateReferenceSystem(out_crs), 'GEOJSON')
        writer.addFeatures(unified_feats)
        del writer

        obj_unified_curb = QgsVectorLayer(output_filepath, "Companies", "ogr")
        field_names = obj_unified_curb.fields().names()
        field_names = [f for f in field_names if ('-' in f and f.startswith("P") and len(f) in (3, 4, 5))]
        values_list = []
        feature_count = obj_unified_curb.featureCount()
        # emptyfields = []
        ic = 0
        df_uni = {}
        for entry in obj_unified_curb.getFeatures():
            dict = {}
            for fname in field_names:
                if entry[fname]:
                    dict[fname] = entry[fname]
                else:
                    dict[fname] = np.NAN
            df_uni[ic] = dict
            ic = ic + 1

        df_uni_df = pd.DataFrame.from_dict(df_uni, "index")
        df_uni_df.index = df_uni_df.index.astype(int)
        emptyfields = df_uni_df.columns[df_uni_df.isnull().all()].tolist()
        emptyfields.extend(['BYLAWS', 'BASE_C_ID'])
        caps = obj_unified_curb.dataProvider().capabilities()
        if caps and QgsVectorDataProvider.DeleteAttributes and len(emptyfields) > 0:
            for fname in emptyfields:
                field_index = obj_unified_curb.fields().indexFromName(fname)
                if field_index != -1:
                    obj_unified_curb.dataProvider().deleteAttributes([field_index])
                    obj_unified_curb.updateFields()

        del obj_unified_curb
        with open(output_filepath) as json_r_file:
            r_json = json.load(json_r_file)
            change_needed = 'manifest' not in r_json
            manifest_tag = {'manifest': {}}
            final_json = {**manifest_tag, **r_json}

        if change_needed:
            with open(output_filepath, 'w') as json_w_file:
                json.dump(final_json, json_w_file)
        if len(idsNotFoundGenSegments):
            feedback.pushInfo("Bylaw segment not having near by general curb segments, Hence not processed--{}".format(
                str(idsNotFoundGenSegments)))
        feedback.pushInfo("********OUTPUT geojson file path****************")
        feedback.pushInfo(output_filepath)
        feedback.pushInfo("************************************************")
        feedback.pushInfo("Unified Curb Layer is generated !!!")
        return {self.OUTPUT: output_filepath}
